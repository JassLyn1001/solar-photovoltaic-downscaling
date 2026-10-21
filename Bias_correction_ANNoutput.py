import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import nc_time_axis
from functools import partial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import copy
from pathlib import Path

# --------------------------------------------------------------------
# STATISTICAL DOWNSCALING: QUANTILE DELTA MAPPING
# --------------------------------------------------------------------
# Qunatile Delta Mapping (Post-Processing) using Normal Distribution

# Create directory list for all CMIP6 models
obs_dir = "/Users/jasslyn/my_data/Graduate_Thesis/data/interim/historical_dataset/"
OBSfname = "era5land_derived_PVOut_potential_JAVA_19700101-20211231.nc"
model_dirs = "/Users/jasslyn/my_data/Graduate_Thesis/data/processed/ANN_output_10climvar"
model_list = os.listdir(model_dirs)

for source_id in model_list:
    print(f"\nProcessing : {source_id}")

    ann_dir = Path('/Users/jasslyn/my_data/Graduate_Thesis/data/processed/ANN_output_10climvar')/source_id
    OBS_path = Path(obs_dir)/OBSfname
    ANN_path = ann_dir/f"ANN_PVout_monthly_{source_id}_historical_197001-201412.nc"

    # Fail early with a clear message if the file doesn't exist
    if not ANN_path.exists():
        print(f"  [SKIP] File not found: {ANN_path}")
        continue

    TARGET_ds = xr.open_dataset(OBS_path)
    ANNout_ds = xr.open_dataset(ANN_path)

    # Calculate monthly mean from observation data
    PV_mean = TARGET_ds.pv_output_potential.resample(valid_time='1ME').mean()
    sel_PV_mean = PV_mean.sel(valid_time=slice('1970','2014')) 
    # Rename reference data dimension for consistency
    PV_target = sel_PV_mean.rename({
        'valid_time': 'time',
        'latitude': 'lat', 
        'longitude': 'lon'
    })

    # Group data by each month for training and evaluation set
    # Select data for training set from 1970-2002 (same period with ANN training set), 
    # and set the rest to val set
    TARGET_train = PV_target.sel(time=slice('1970', '2002')).groupby("time.month")
    INPUT_train = ANNout_ds.PVout.sel(time=slice('1970', '2002')).groupby("time.month")
    INPUT_eval = ANNout_ds.PVout.sel(time=slice('2003', '2014')).groupby("time.month")

    def QDM_norm(target_train, input_train, input_eval, min_n_eval=10):
        """
        Quantile Delta Mapping (additive, parametric Normal) -- Cannon et al. (2015).
    
        min_n_eval: jumlah minimum sampel valid untuk fitting distribusi periode
        eval. DEFAULT=10 karena periode eval sekarang mencakup 2003-2014
        (val+test digabung, n=12 tahun/bulan) -- jauh lebih stabil dibanding n=6
        (val saja) yang dipakai sebelumnya. Ambang 10 (bukan 12 persis) memberi
        sedikit toleransi kalau ada 1-2 tahun NaN per grid/bulan tanpa langsung
        menggagalkan seluruh koreksi grid tsb.
        """
        target_train = np.asarray(target_train, dtype=float)
        input_train = np.asarray(input_train, dtype=float)
        input_eval = np.asarray(input_eval, dtype=float)
    
        # Return nan for ocean grid
        if np.all(np.isnan(target_train)) or np.all(np.isnan(input_train)):
            return np.full(len(input_eval), np.nan)
    
        # Apply the calibration only for overlap obs & base (TRAIN period)
        n_overlap = min(len(target_train), len(input_train))
        target_train_cal, input_train_cal = target_train[:n_overlap], input_train[:n_overlap]
        valid_both = ~np.isnan(target_train_cal) & ~np.isnan(input_train_cal)
    
        if np.sum(valid_both) < 33:  # 33 years for 1970-2002 (training/calibration set)
            return np.full(len(input_eval), np.nan)
    
        target_train_clean = target_train_cal[valid_both]
        input_train_clean = input_train_cal[valid_both]
    
        # ---- BARU (QDM): cek validitas sampel input_eval sebelum fitting -----
        valid_eval = ~np.isnan(input_eval)
        if np.sum(valid_eval) < min_n_eval:
            return np.full(len(input_eval), np.nan)
        input_eval_clean = input_eval[valid_eval]
        # ------------------------------------------------------------------
    
        try:
            # Calculate parameters of NORMAL distribution (loc/mean, scale/std)
            # -- TRAIN target & TRAIN input, sama seperti QM standar
            loc_target, scale_target = stats.norm.fit(target_train_clean, method="MLE")
            loc_input, scale_input = stats.norm.fit(input_train_clean, method="MLE")
    
            # ---- BARU (QDM): fit distribusi Normal untuk periode EVAL juga ----
            # Inilah inti perbedaan QDM -- QM standar tidak pernah fit distribusi
            # input_eval; QDM memerlukannya untuk menentukan tau (kuantil) setiap
            # nilai input_eval relatif terhadap sebarannya sendiri.
            loc_input_eval, scale_input_eval = stats.norm.fit(input_eval_clean, method="MLE")
            # ---------------------------------------------------------------
    
            if not np.all(np.isfinite([loc_target, scale_target,
                                        loc_input, scale_input,
                                        loc_input_eval, scale_input_eval])):
                return np.full(len(input_eval), np.nan)
            if scale_input_eval <= 0:  # guard pembagian oleh nol/negatif
                return np.full(len(input_eval), np.nan)
    
            # ---- BARU (QDM): hitung tau per titik dari distribusi EVAL -------
            # (QM standar TIDAK melakukan ini -- QM standar memakai transfer
            # function tunggal berbasis regresi pada grid probabilitas tetap,
            # diterapkan sama rata ke seluruh input_eval)
            tau = stats.norm.cdf(input_eval_clean, loc_input_eval, scale_input_eval) # CDF
            tau = np.clip(tau, 0.001, 0.999)  # hindari tau di batas ekstrem (0/1 -> inf)
    
            # Petakan tau melalui distribusi TRAIN target -> "QM analog"
            x_qm = stats.norm.ppf(tau, loc_target, scale_target)
    
            # Petakan tau yang SAMA melalui distribusi TRAIN input -> "historical
            # analog": nilai setara yang akan muncul di periode TRAIN pada
            # kuantil yang sama dengan nilai eval ini
            x_hist_analog = stats.norm.ppf(tau, loc_input, scale_input)
    
            # Delta = seberapa jauh nilai eval asli menyimpang dari historical
            # analog-nya -- inilah sinyal "perubahan" periode eval yang
            # dipertahankan (bukan ditimpa) oleh QDM
            delta = input_eval_clean - x_hist_analog
    
            # Nilai akhir = QM analog + delta (ADDITIVE QDM)
            corrected_clean = x_qm + delta
            # -------------------------------------------------------------
    
            # Kembalikan ke ukuran asli input_eval (isi NaN untuk titik tak valid)
            input_corrected = np.full(len(input_eval), np.nan)
            input_corrected[valid_eval] = corrected_clean
            return input_corrected
    
        except Exception:
            return np.full(len(input_eval), np.nan)
    
    
    # ============================================================================
    # QM standar untuk periode TRAIN (1970-2002) -- fungsi asli Anda, tidak diubah
    # ============================================================================
    def QM_norm(target_train, input_train, input_eval):
        """Quantile Mapping standar (dipakai utk mengoreksi periode TRAIN itu
        sendiri -- periode kalibrasi, bukan periode eval terpisah)."""
        target_train = np.asarray(target_train, dtype=float)
        input_train = np.asarray(input_train, dtype=float)
        input_eval = np.asarray(input_eval, dtype=float)
    
        if np.all(np.isnan(target_train)) or np.all(np.isnan(input_train)):
            return np.full(len(input_eval), np.nan)
    
        n_overlap = min(len(target_train), len(input_train))
        target_train_cal, input_train_cal = target_train[:n_overlap], input_train[:n_overlap]
        valid_both = ~np.isnan(target_train_cal) & ~np.isnan(input_train_cal)
    
        if np.sum(valid_both) < 33:
            return np.full(len(input_eval), np.nan)
    
        target_train_clean = target_train_cal[valid_both]
        input_train_clean = input_train_cal[valid_both]
    
        try:
            loc_target, scale_target = stats.norm.fit(target_train_clean, method="MLE")
            loc_input, scale_input = stats.norm.fit(input_train_clean, method="MLE")
    
            if not np.all(np.isfinite([loc_target, scale_target, loc_input, scale_input])):
                return np.full(len(input_eval), np.nan)
    
            prob_grid = np.linspace(0.01, 0.99, 100)
            target_norm_invcdf = stats.norm.ppf(prob_grid, loc_target, scale_target)
            input_norm_invcdf = stats.norm.ppf(prob_grid, loc_input, scale_input)
    
            valid = (np.isfinite(target_norm_invcdf) & np.isfinite(input_norm_invcdf)
                    & (input_norm_invcdf > 0) & (target_norm_invcdf > 0))
            if np.sum(valid) < 10:
                return np.full(len(input_eval), np.nan)
    
            target_norm_invcdf, input_norm_invcdf = target_norm_invcdf[valid], input_norm_invcdf[valid]
            coeffs = np.polyfit(input_norm_invcdf, target_norm_invcdf, 1)
            slope, intercept = coeffs[0], coeffs[1]
    
            return intercept + slope * input_eval
        except Exception:
            return np.full(len(input_eval), np.nan)
    
    
    # ============================================================================
    # Terapkan QM (train, 1970-2002) dan QDM (eval, 2003-2014) secara terpisah
    # ============================================================================
    corrected_train = {}
    corrected_eval = {}
    for month in range(1, 13):
        target_train_m = TARGET_train[month]
        input_train_m = INPUT_train[month]
        input_eval_m = INPUT_eval[month]
    
        # -- QM standar: koreksi periode TRAIN itu sendiri (input_eval=input_train) --
        qm_train = xr.apply_ufunc(
            QM_norm,
            target_train_m, input_train_m, input_train_m,  # input_eval = input_train
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[['time']],
            exclude_dims={'time'},
            vectorize=True,
            output_dtypes=[np.float32])
        qm_train['time'] = input_train_m.time
        corrected_train[month] = qm_train.transpose('time', ...)
    
        # -- QDM: koreksi periode EVAL (2003-2014) --
        qdm_eval = xr.apply_ufunc(
            QDM_norm,
            target_train_m, input_train_m, input_eval_m,
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[['time']],
            exclude_dims={'time'},
            vectorize=True,
            output_dtypes=[np.float32])
        qdm_eval['time'] = input_eval_m.time
        corrected_eval[month] = qdm_eval.transpose('time', ...)
    
        print(f'QM (train) + QDM (eval) untuk bulan {month} selesai')
    
    # Gabungkan tiap bulan, lalu gabungkan train+eval jadi satu deret 1970-2014
    ANN_QM_train = xr.concat([corrected_train[m] for m in range(1, 13)],
                            dim='time').sortby('time')
    ANN_QDM_eval = xr.concat([corrected_eval[m] for m in range(1, 13)],
                            dim='time').sortby('time')
    
    # Hasil akhir: QM (1970-2002) + QDM (2003-2014), satu deret waktu utuh
    ANN_biascorrected = xr.concat([ANN_QM_train, ANN_QDM_eval], dim='time').sortby('time')

    # Get attributes from all coordinate dimensions
    time_att = PV_target.time.attrs 
    lon_att = PV_target.lon.attrs
    lat_att = PV_target.lat.attrs

    # Define a function to assign dataarray attributes for bias-corrected data
    def assign_dim_attrs(da):
        da['time'] = da.time.assign_attrs(time_att)
        da['lat'] = da.lat.assign_attrs(lat_att)
        da['lon'] = da.lon.assign_attrs(lon_att)
        return da

    # Define dataarray's attributes
    da_attrs = {
        'standard_name': 'pv_energy_potential',
        'long_name': 'Monthly PV energy potential',
        'units': 'kWh m-2 month-1',
        }
    
    ds_attrs_anncdf = {
        'title': 'Statistical Downscaling output based on hybrid approach (ANN combined with Quantile Delta Mapping)',
        'authors': 'Jassica Listyarin',
        'institution': 'Department of Geophysics and Meteorology, Faculty of Mathematics and Natural Sciences, IPB University',
        'table_id': 'mon',
    }
    VAR = 'PVout'
    # Assign attributes for bias-corrected baseline data
    ANNCDF_attrs = assign_dim_attrs(ANN_biascorrected)
    ANNCDF_da_attrs = ANNCDF_attrs.assign_attrs(da_attrs)
    ANNCDF_ds = ANNCDF_da_attrs.to_dataset(dim=None, name=f'{VAR}')
    ANNCDF_ds= ANNCDF_ds.assign_attrs(ds_attrs_anncdf)

    # Let's quick check for the final results
    #ANNCDF_ds.PVout[-1,:,:].plot()
    #ANNCDF_ds.PVout[:,25,50].plot()
    #PV_target[:,25,50].plot()

    outdir_anncdf = f'/Users/jasslyn/my_data/Graduate_Thesis/data/processed/ANNQDM/'# Define file's configurations
    path_anncdf = Path(outdir_anncdf)/source_id
    try:
        path_anncdf.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{source_id}' is created or already exists")
    except Exception as e:
        print(f"Failed to create directory: {str(e)}")
        raise
    period = '197001-201412'
    table_id = 'mon'

    def save_final_dataset(ds):
        """Helper function to save datasets into netcdf format with error handling"""
        try:
            filename = f"ANNCDF_{VAR}_monthly_{source_id}_historical_{period}.nc"
            output_path = path_anncdf / filename
            # Add another attributes
            ds.attrs['experiment_id'] = 'historical'
            ds.attrs['frequency'] = table_id
            ds.attrs['mip_era'] = 'CMIP6'
            ds.attrs['source_id'] = source_id

            print(f"Attempting to save to: {output_path}")

            # Load and write in one operation to minimize memory usage
            ds.to_netcdf(
                output_path,
                format="NETCDF4",
                engine="netcdf4",
                encoding={VAR: {"dtype": "float32"}, 
                        'time': {'dtype': 'float64', 
                                'units': 'days since 1850-01-01 00:00:00'}},  # Use variable name from VAR
                unlimited_dims=['time']
            )
            print(f"Successfully saved: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")
            raise

    # Save bias-corrected data to new netcdf file
    save_final_dataset(ANNCDF_ds)
    print(f"{source_id} - DONE")

print(f"Bias-correction process for all {len(model_list)} models is done")
 


