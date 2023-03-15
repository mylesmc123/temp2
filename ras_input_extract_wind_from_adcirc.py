#!/usr/bin/env python3

from datetime import datetime, timedelta, timezone
from logging import raiseExceptions
from tqdm import tqdm
import os
import numpy as np

def validateBounds(parser, bounds_nsew):
    # validation of bounds will restricts the bound to a northwest hemisphere, but vaues are givent a northing and easting.
    # Meaning the east and west values will always be negative"
    if not 0 <= bounds_nsew[0] <=90:
        parser.error("the North bound is not in the Northern Hemisphere and given a northing value. Value should be between 0 and 90.")
    if not 0 <= bounds_nsew[1] <=90:
        parser.error("the South bound is not in the Northern Hemisphere. Value should be between 0 and 90.")
    if not bounds_nsew[0] > bounds_nsew[1]:
        parser.error("the North bound must be greater than the South bound.")
    if not -180 <= bounds_nsew[2] <= 0:
        parser.error("the East bound is not in the Western Hemisphere and given as an easting value. Value should be between -180 and 0.")
    if not -180 <= bounds_nsew[3] <= 0:
        parser.error("the West bound is not in the Western Hemisphere and given as an easting value. Value should be between -180 and 0.")
    if not bounds_nsew[2] > bounds_nsew[3]:
        parser.error("the East bound must be greater than the West bound. \
            Since these values are negative in the western hemisphere and required to be input here as an easting (a negative number), \
            the East bound must be closer to 0.")

def createPointFile(cellsize, bounds_nsew, pointsFile):
    x = np.arange(bounds_nsew[3], bounds_nsew[2], cellsize)
    y = np.arange(bounds_nsew[0], bounds_nsew[1], cellsize)
    length_values = len(x) * len(y)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.empty((length_values, 2))
    points[:, 0] = x_grid.flatten()
    points[:, 1] = y_grid.flatten()
    np.savetxt(pointsFile, points, delimiter=',', fmt='%1.2f', comments='')
    return pointsFile
class AdcircExtract:
    def __init__(self, filename: str, pointfile: str, coldstart: datetime):
        import netCDF4 as nc

        self.__filename = filename
        self.__ncfile = nc.Dataset(self.__filename)
        self.__pointfile = pointfile
        self.__coldstart = coldstart
        self.__nnode = None
        self.__nelem = None
        self.__nodes = None
        self.__elements = None
        self.__centroid = None
        self.__tree = None
        self.__variables = []
        self.__units = []
        self.__datums = []
        self.__standard_names = []
        self.__long_names = []
        self.__extract_points = None
        self.__n_stations = None
        self.__point_indices = None
        print ("Reading Mesh...")
        self.__read_mesh()
        # self.__find_variable()
        print ("Setting Variable...")
        self.__set_variable()
        # self.__set_wind_variables()
        print ("Getting Dimensions...")
        self.__get_dimensions()
        print ("Reading Points...")
        self.__read_points()
        print ("Find Point Indices...")
        self.__find_point_indices()

    def __is_inside(self, x, y, element_index) -> bool:
        import math
        import numpy as np

        nodes = self.__elements[element_index]
        n1 = nodes[0] - 1
        n2 = nodes[1] - 1
        n3 = nodes[2] - 1
        xx = np.array((self.__nodes[0][n1], self.__nodes[0][n2], self.__nodes[0][n3]))
        yy = np.array((self.__nodes[1][n1], self.__nodes[1][n2], self.__nodes[1][n3]))

        s0 = abs(
            (xx[1] * yy[2] - xx[2] * yy[1])
            - (x * yy[2] - xx[2] * y)
            + (x * yy[1] - xx[1] * y)
        )
        s1 = abs(
            (x * yy[2] - xx[2] * y)
            - (xx[0] * yy[2] - xx[2] * yy[0])
            + (xx[0] * y - x * yy[0])
        )
        s2 = abs(
            (xx[1] * y - x * yy[1])
            - (xx[0] * y - x * yy[0])
            + (xx[0] * yy[1] - xx[1] * yy[0])
        )
        tt = abs(
            (xx[1] * yy[2] - xx[2] * yy[1])
            - (xx[0] * yy[2] - xx[2] * yy[0])
            + (xx[0] * yy[1] - xx[1] * yy[0])
        )
        return s0 + s1 + s2 <= tt

    def __interpolation_weight(self, x, y, element_index):
        import numpy as np

        nodes = self.__elements[element_index]
        n1 = nodes[0] - 1
        n2 = nodes[1] - 1
        n3 = nodes[2] - 1
        xx = np.array((self.__nodes[0][n1], self.__nodes[0][n2], self.__nodes[0][n3]))
        yy = np.array((self.__nodes[1][n1], self.__nodes[1][n2], self.__nodes[1][n3]))

        denom = (yy[1] - yy[2]) * (xx[0] - xx[2]) + (xx[2] - xx[1]) * (yy[0] - yy[2])
        w0 = ((yy[1] - yy[2]) * (x - xx[2]) + (xx[2] - xx[1]) * (y - yy[2])) / denom
        w1 = ((yy[2] - yy[0]) * (x - xx[2]) + (xx[0] - xx[2]) * (y - yy[2])) / denom
        w2 = 1.0 - w1 - w0
        return [n1, n2, n3, w0, w1, w2]

    def __find_point_indices(self):
        self.__point_indices = []
        for p in tqdm(self.__extract_points):
            _, idx = self.__tree.query([p[0], p[1]], k=10)
            found = False
            for d in idx:
                found = self.__is_inside(p[0], p[1], d)
                if found:
                    self.__point_indices.append(
                        self.__interpolation_weight(p[0], p[1], d)
                    )
                    break
            if not found:
                self.__point_indices.append([-9999, -9999, -9999, 0, 0, 0])

    def __read_points(self):
        import csv

        self.__extract_points = []
        with open(self.__pointfile, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # print (row)
                x = float(row[0])
                y = float(row[1])
                # tag = int(row[2])
                # self.__extract_points.append([x, y, tag])
                self.__extract_points.append([x, y])
        self.__n_stations = len(self.__extract_points)

    def __set_variable(self):
        candidate_variables = ["windx", "windy"]
        units = "m s-1"
        datum = "n/a"
        standard_name = ["eastward_wind", "northward_wind"]
        long_name = ["e/w wind velocity", "n/s wind velocity"]
        print (str(len(candidate_variables)))
        for i in range(len(candidate_variables)):
            if candidate_variables[i] in self.__ncfile.variables:
                self.__variables.append(candidate_variables[i])
                self.__units.append(units)
                self.__datums.append(datum)
                self.__standard_names.append(standard_name[i])
                self.__long_names.append(long_name[i])
            else: raise RuntimeError(f"Could not locate {candidate_variables[i]} in self.__ncfile.variables")
        return
        
    def __set_wind_variables(self):
        candidate_variables = ["windx", "windy"]
        units = "m s-1"
        datum = "n/a"
        standard_name = ["eastward_wind", "northward_wind"]
        long_name = ["e/w wind velocity", "n/s wind velocity"]
        print (str(len(candidate_variables)))
        
        if candidate_variables[0] in self.__ncfile.variables:
            self.__windx_variable = candidate_variables[0]
            self.__windx_units = units[0]
            self.__windx_datum = datum[0]
            self.__windx_standard_name = standard_name[0]
            self.__windx_long_name = long_name[0]
        else: raise RuntimeError("Could not locate a valid ADCIRC variable")

        if candidate_variables[1] in self.__ncfile.variables:
            self.__windy_variable = candidate_variables[1]
            self.__windy_units = units[1]
            self.__windy_datum = datum[1]
            self.__windy_standard_name = standard_name[1]
            self.__windy_long_name = long_name[1]
        else: raise RuntimeError("Could not locate a valid ADCIRC variable")
        
        return
        

    def __find_variable(self):
        candidate_variables = [
            "zeta",
            "swan_HS",
            "swan_TPS",
            "swan_TM01",
            "swan_DIR",
            "zeta_max",
            "swan_HS_max",
            "swan_TPS_max",
            "swan_TM01_max",
            "swan_DIR_max",
        ]
        units = ["m", "m", "s", "s", "deg", "m", "m", "s", "s", "deg"]
        datum = [
            "navd88 2009.55",
            "m",
            "n/a",
            "n/a",
            "n/a",
            "navd88 2009.55",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
        ]
        standard_name = [
            "sea_surface_height_above_geoid",
            "sea_surface_wave_significant_height",
            "smoothed peak period",
            "mean wave direction",
            "maximum water surface elevationabove geoid",
            "maximum significant wave height",
            "maximum smoothed peak period",
            "maximum TM01 mean wave period",
            "maximum mean wave direction",
        ]
        long_name = [
            "water surface elevation above geoid",
            "significant wave height",
            "sea_surface_wave_period_at_variance_spectral_density_maximum",
            "sea_surface_wave_to_direction",
            "maximum_sea_surface_height_above_geoid",
            "maximum_sea_surface_wave_significant_height",
            "maximum_sea_surface_wave_period_at_variance_spectral_density_maximum",
            "maximum_sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
            "maximum_sea_surface_wave_to_direction",
        ]

        for i in range(len(candidate_variables)):
            if candidate_variables[i] in self.__ncfile.variables:
                self.__variable = candidate_variables[i]
                self.__units = units[i]
                self.__datum = datum[i]
                self.__standard_name = standard_name[i]
                self.__long_name = long_name[i]
                return
        raise RuntimeError("Could not locate a valid ADCIRC variable")

    def __get_dimensions(self):
        self.__n_step = self.__ncfile.dimensions["time"].size

    def __read_mesh(self):
        import numpy as np
        from scipy.spatial import KDTree

        x = self.__ncfile["x"][:]
        y = self.__ncfile["y"][:]
        self.__nodes = [x, y]
        self.__elements = self.__ncfile["element"][:]
        self.__nelem = self.__elements.shape[0]
        print (f"Number of Mesh Elements {str(self.__nelem)}")
        self.__nnode = x.shape[0]
        self.__centroid = np.zeros((self.__elements.shape[0], 2), dtype=float)

        for i in tqdm(range(self.__nelem)):
            nodes = self.__elements[i]
            n1 = nodes[0] - 1
            n2 = nodes[1] - 1
            n3 = nodes[2] - 1
            x_c = (
                self.__nodes[0][n1] + self.__nodes[0][n2] + self.__nodes[0][n3]
            ) / 3.0
            y_c = (
                self.__nodes[1][n1] + self.__nodes[1][n2] + self.__nodes[1][n3]
            ) / 3.0
            self.__centroid[i][0] = x_c
            self.__centroid[i][1] = y_c

        print("Create KDTree")
        self.__tree = KDTree(self.__centroid)

    @staticmethod
    def __normalize_weights(z0, z1, z2, w0, w1, w2):
        if z0 > -999 and z1 > -999 and z2 > -999:
            return w0, w1, w2
        elif z0 < -999 and z1 > -999 and z2 > -999:
            f = 1.0 / (w1 + w2)
            w0 = 0.0
            w1 *= f
            w2 *= f
        elif z0 > -999 and z1 < -999 and z2 > -999:
            f = 1.0 / (w0 + w3)
            w1 = 0.0
            w0 *= f
            w2 *= f
        elif z0 > -999 and z1 > -999 and z2 < -999:
            f = 1.0 / (w0 + w1)
            w0 *= f
            w1 *= f
            w2 = 0.0
        elif z0 > -999 and z1 < -999 and z2 < -999:
            w0 = 1.0
            w1 = 0.0
            w2 = 0.0
        elif z0 < -999 and z1 > -999 and z2 < -999:
            w0 = 0.0
            w1 = 1.0
            w2 = 0.0
        elif z0 < -999 and z1 < -999 and z2 > -999:
            w0 = 0.0
            w1 = 0.0
            w2 = 1.0
        else:
            w0 = 1.0
            w1 = 1.0
            w2 = 1.0
        return w0, w1, w2

    def extract(self, temp_output_file):
        import numpy as np
        import netCDF4 as nc

        time = np.zeros((self.__n_step), dtype=int)
        data = np.zeros((self.__n_step, self.__n_stations), dtype=float)
        all_times = self.__ncfile["time"][:]
        
        ds = nc.Dataset(temp_output_file, "w", format="NETCDF4")
        time_dim = ds.createDimension("time", self.__n_step)
        station_dim = ds.createDimension("nstation", self.__n_stations)

        global_attrs = {
            "Conventions": "CF-1.6,UGRID-0.9",
            "title": "ADCIRD Wind Data, HEC-RAS Format",
            "institution": "The Water Institute",
            "source": "LFFS ADCIRC Output",
            "history": datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            "references": "https://github.com/adcirc/MetGet",
            "metadata_conventions": "Unidata Dataset Discovery v1.0",
            "summary": "Data generated from ADCIRC fort.74.nc and wind data converted to a new netCDF for use in HEC-RAS",
            "date_created": datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            }
        ds.setncatts(global_attrs)

        timevar = ds.createVariable("time", "f8", ("time"))
        # "minutes since 2021-07-27 13:00:00.0 +0000"
        # --coldstart", "2021-07-27 12:00:00"
        # timevar.units = "minutes since 2021-07-27 13:00:00.0 +0000"
        coldstart_str = str(self.__coldstart).split("+")[0]
        timevar.units = f"minutes since {coldstart_str}.0 +0000"

        # timevar.base_date = "1970-01-01 00:00:00"
        # timevar.calendar = "gregorian"
        timevar.standard_name = "time"
        timevar.long_name = "time"
        timevar.axis = "T";

        crs = ds.createVariable("crs", "i8")
        crs.long_name = "coordinate reference system"
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.crs_wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]"
        crs.proj4_params = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        crs.epsg_code = "EPSG:4326"

        lat = ds.createVariable("latitude", "f8", ("nstation"), zlib=True, complevel=2)
        lat.reference = "EPSG:4326"
        lat.standard_name = "latitude"
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        lat.axis = "Y"
       
        lon = ds.createVariable("longitude", "f8", ("nstation"), zlib=True, complevel=2)
        lon.reference = "EPSG:4326"
        lon.standard_name = "longitude"
        lon.long_name = "Longitude"
        lon.units = "degrees_east"
        lon.axis = "X"
        # tag = ds.createVariable("point_type", "i4", ("nstation"), zlib=True, complevel=2
        # )
        # tag.types = "0=gate, 1=levee, 2=roadway"

        for idx,var in enumerate(self.__variables):
            print(f"Extracting points for each timestep for {self.__variables[idx]}...")
            for i in tqdm(range(self.__n_step)):
                record = self.__ncfile[var][i]
                # From cold start time from command line argument: add alltimes[i].
                # Cold start time should match whats in the ADCIRC output reference time <-- Nocheck on this performed. User entered reference only.
                sim_time = self.__coldstart + timedelta(seconds=all_times[i])
                # Convert to epoch time: seconds since 1970-01-01 00:00:00
                # time[i] = datetime.timestamp(sim_time)
                # Convert to minutes since coldstart ref time
                time[i] = all_times[i]/60
                for j in range(self.__n_stations):
                    z0 = record[self.__point_indices[j][0]]
                    z1 = record[self.__point_indices[j][1]]
                    z2 = record[self.__point_indices[j][2]]
                    w0, w1, w2 = AdcircExtract.__normalize_weights(
                        z0,
                        z1,
                        z2,
                        self.__point_indices[j][3],
                        self.__point_indices[j][4],
                        self.__point_indices[j][5],
                    )
                    data[i][j] = z0 * w0 + z1 * w1 + z2 * w2
            print(f"create netCDF dataset variable for {self.__variables[idx]}...")
            datavar = ds.createVariable(
                self.__variables[idx],
                "f8",
                ("time", "nstation"),
                fill_value=-99999,
                zlib=True,
                complevel=2,
            )
            datavar.adcirc_type = self.__variables[idx]
            datavar.standard_name = self.__standard_names[idx]
            datavar.long_name = self.__long_names[idx]
            datavar.units = self.__units[idx]
            datavar.datum = self.__datums[idx]
            datavar.grid_mapping = "crs" 

            datavar[:, :] = data

        print("Set extracted points to Lat/Lon")
        for i in tqdm(range(self.__n_stations)):
            lon[i] = self.__extract_points[i][0]
            lat[i] = self.__extract_points[i][1]
            # tag[i] = self.__extract_points[i][2]

        timevar[:] = time
    
    def unstack(self, output_file, temp_output_file):
        import xarray as xr

        ds = xr.open_dataset(temp_output_file, decode_cf=False)
        ds = ds.assign_coords(({
                "latitude": ("nstation", ds.latitude.data),
                "longitude": ("nstation", ds.longitude.data),                    
        }))
        ds = ds.reindex()
        ds = ds.set_index(nstation=["latitude","longitude"])
        ds = ds.unstack(["nstation"])
        ds.longitude.attrs = {
            "long_name" : "Longitude",
            "units" : "degrees_east",
            "axis" : "X"
        }
        ds.latitude.attrs = {
            "long_name" : "Latitude",
            "units" : "degrees_north",
            "axis" : "Y"
        }
        ds.to_netcdf(path=output_file, format="NETCDF4", engine="netcdf4")
        print(f"Output File Created: {output_file}")

def main():
    import argparse

    p = argparse.ArgumentParser(description="Point extraction for ADCIRC time series")
    p.add_argument(
        "--file", help="Name of the file to extract from", required=True, type=str
    )

    p.add_argument(
        "--cellsize",
        help="Cell size of structured grid in degrees. Warning: Too small of a Cell Size will cause memory errors. \
            A value of 0.05 used for the USACE coastal model.",
        required=False,
        type=float
    )

    p.add_argument(
        "--bounds_nsew",
        help="Bounds should entered in a sequence of North, South, East West and include negative coordinates to indicate western hemisphere \
             (I.E. for the Louisiana Coastal model: 30.7 28.3 -88 -95). Either provide --points or [--bounds_nsew & --cellsize argument].",
        required=False,
        nargs=4,
        type=float
    )

    p.add_argument(
        "--points",
        help="Name of point file for extracted locations. Either provide --points or [--bounds_nsew & --cellsize argument].",
        required=False,
        type=str,
    )
    p.add_argument(
        "--coldstart",
        help="Cold start time for ADCIRC simulation",
        required=True,
        type=datetime.fromisoformat,
    )
    p.add_argument(
        "--output", help="Name of output file to create", required=True, type=str
    )

    args = p.parse_args()

    # ...The user is sending UTC, so make python do the same
    coldstart_utc = datetime(
        args.coldstart.year,
        args.coldstart.month,
        args.coldstart.day,
        args.coldstart.hour,
        args.coldstart.minute,
        0,
        
    tzinfo=timezone.utc,
    )

    # Use either user provided user points file or create one via bounds argument. Validate bounds in the process.
    if (args.points is not None) & (args.bounds_nsew is not None):
        p.error("Both --points and --bounds_nsew provided. Either provide --points or [--bounds_nsew & --cellsize arguments].")
    elif (args.bounds_nsew is not None) & (args.cellsize is None):
        p.error("--bounds_nsew requires --cellsize argument to also be given.")
    elif (args.bounds_nsew is None) & (args.cellsize is not None):
        p.error("--cellsize requires --bounds_nsew argument to also be given.")
    elif (args.bounds_nsew is None) & (args.points is None):
        p.error("Either provide --points or [--bounds_nsew & --cellsize arguments].")
    elif (args.bounds_nsew is not None) & (args.cellsize is not None):
        # Bounds given, Validate Bounds, Create pointFile.
        validateBounds(p, args.bounds_nsew)
        pointsFile = 'pointFile_WindGrid.txt'
        args.points = createPointFile(args.cellsize, args.bounds_nsew, pointsFile)
         
    print("Begin Extracting Timeseries...")
    temp_output_file = args.output.split(".")[0]+"_temp.nc"
    extractor = AdcircExtract(args.file, args.points, coldstart_utc)
    extractor.extract(temp_output_file)
    extractor.unstack(args.output, temp_output_file)
    
    # Remove temp files
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
        print ("temp output nc file removed")
    if os.path.exists(pointsFile):
        os.remove(pointsFile)
        print ("temp output points file removed")

if __name__ == "__main__":
    main()
