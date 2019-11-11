#### clean_line_breaks.py
#### November 1, 2019
#### PP275 -- Lab 7
#### Contact email: jsayre@berkeley.edu


def clean_line_breaks(df):
    '''clean_line_breaks cleans up noncontiguous polylines in ZAF railroads
    Inputs: geopandas dataframe of ZAF railroad shapefile
    Outputs: cleaned railroad gpd dataframe'''
    from shapely.geometry import LineString
    for i in range(len(df)):
        try:
            line_lats, line_lons = df['geometry'][i].xy
        except:
            if i == 481: ### for whatever reason this one has a different structure
                df.loc[i,'geometry'] = df['geometry'][i].geoms[0]
            else:
                full_lats, full_lons = [], []
                for geom in df['geometry'][i].geoms:
                    line_lats, line_lons = geom.xy
                    full_lats.extend(line_lats)
                    full_lons.extend(line_lons)
                df.loc[i,'geometry'] = LineString(list(zip(full_lats,full_lons)))
    return df

