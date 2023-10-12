
stations = ['ABZ', 'AMM', 'ASP', 'BEI', 'BOT', 'BSL', 'EGB', 'FKL', 'HAD', 
            'HEL', 'HPB', 'HYY', 'KCE', 'KPZ', 'MAR', 'MHD', 'MLP', 'MUK',
            'NAN', 'NEU', 'POV', 'SAO', 'SCH', 'SGP', 'UAE', 'PRL',
            'VAR', 'VHL', 'VIE', 'WAL', 'ZOT']

aerosols = ['aermr01', 'aermr02', 'aermr03', 'aermr04', 'aermr05', 'aermr06', 
            'aermr07', 'aermr08', 'aermr09', 'aermr10', 'aermr11']

gases = ['co', 'c5h8', 'c10h16', 'nh3', 'no', 'no2', 'so2']

met = ['d2m', 't2m', 'blh', 'rh', 'wind_speed', 'crwc']
        
loc = ['latitude', 'longitude', 'altitude']

env_types = ['env_type_urban', 'env_type_urban_background', 'env_type_rural', 'env_type_rural_background', 'env_type_remote']

feature_list = gases + met + aerosols + loc + env_types