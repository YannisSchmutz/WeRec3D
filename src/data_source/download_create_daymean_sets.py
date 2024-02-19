import cdsapi
import subprocess
import os
import click

from src.config import DL_LAT_START, DL_LAT_END, DL_LON_START, DL_LON_END


VAR_NAMES = ['2m_temperature', 'mean_sea_level_pressure']
YEARS = [str(x) for x in range(1950, 2024)]
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
DAYS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
TIMES = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
AREA = [DL_LAT_START, DL_LON_START, DL_LAT_END, DL_LON_END]

VAR_MAP = {'2m_temperature': 't2m',
           'mean_sea_level_pressure': 'msl'}

URL = "https://cds.climate.copernicus.eu/api/v2"


@click.command()
@click.option('-k', '--key', required=True, type=str)
def main(key):
    """

    :param key: uid:key
    :return:
    """
    download_file_name = 'download.nc'
    client = cdsapi.Client(url=URL, key=key)

    for var in VAR_NAMES:
        for year in YEARS:
            print(f"Going to download v={var}, y={year}")
            request_conf = {'product_type': 'reanalysis',
                            'variable': var,
                            'year': year,
                            'month': MONTHS,
                            'day': DAYS,
                            'time': TIMES,
                            'area': AREA,
                            'format': 'netcdf',
                            }
            client.retrieve('reanalysis-era5-single-levels', request_conf, download_file_name)
            print("Download completed!")

            print('Calculating daymean!')
            subprocess.run(["cdo", "daymean", download_file_name, f"daymean_sets/era5_daymean_{var}_{year}.nc"])
            print("Daymean calculation completed!")

            if os.path.isfile(download_file_name):
                os.remove(download_file_name)


if __name__ == '__main__':
    main()
