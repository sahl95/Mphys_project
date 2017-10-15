import glob
import os
import sys
from bs4 import BeautifulSoup
from unidecode import unidecode
import requests
import numpy as np
import pandas as pd
from scipy import stats
import numpy.ma as ma

def mean_data(obj_id):
    obj_id_split = obj_id.split(' ')
    obj_id = obj_id_split[0]
    output_id = obj_id_split[0]
    for i in range(1, len(obj_id_split)):
        obj_id += '+'+obj_id_split[i]
        output_id += '_'+obj_id_split[i]

    # print(obj_id)
    # obj_id = 'HD 108874'
    # a, b = obj_id.split(' ')
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/ExoOverview/nph-ExoOverview?objname={}&type=&label&aliases&exo&iden&orb&ppar&tran&note&disc&ospar&ts&nalc&force=&dhxr1507830887922".format(obj_id)
    print('Accessing '+url+'...\n')
    response = requests.get(url)

    bs = BeautifulSoup(response.content, "html.parser")

    for idx, title in enumerate(bs.findAll('div', {'class': 'data'})):
        name = title.find('th').text
        if name == 'Planet Orbital Properties':
            index = idx
            # print(name, idx)

    planet_props = bs.findAll('div', {'class': 'data'})

    column_names = []
    planets = []
    p_idx = 0

    for idx, text in enumerate(planet_props[index].findAll()):
        text = unidecode(str(text))

        # print(te)
        if 'th' in text:
            if 'class' not in text:
                if 'Reference' not in text:
                    # if 'td' not in text:
                    if 'href' not in text:
                        column_names.append(text[4:-5])
                # else:
                #     column_names.append(text.split('\n')[-1][4:-5])
                        # print(text[4:-5])
        if idx > 1:
            text_split = text.split('\n')
            for t in text_split:
                if 'td' in t:
                    if 'href' not in t:
                        # if 'class' not in t:
                        t = t[4:-5]
                        # print(t)
                        if '+-' in t:
                            t = t.split('+-')[0].split(' ')[-1]
                            planets[p_idx-1].append(t)
                        elif 'lt' in t or 'gt' in t:
                            t = t.split(';')[-1]
                            planets[p_idx-1].append(t)
                        else:
                            t = t.split('span')
                            if len(t) == 1:
                                t = t[0].split(' ')[-1]
                                if 'null' not in t:
                                    if t.isdigit() or '.' in t:
                                        planets[p_idx-1].append(t)
                                    else:
                                        # print(t)
                                        if len(t) == 1:
                                            planets.append([])
                                            planets[p_idx].append(t)
                                            p_idx += 1
                                else:
                                    planets[p_idx-1].append(t)
                                # print()
                            else:
                                t = t[1].split('>')[1].split('<')[0]
                                planets[p_idx-1].append(t)
                            # print(t)
    planets = planets[::2]

    column_names_2 = []
    planets_2 = []
    p_idx = 0

    for idx, title in enumerate(bs.findAll('div', {'class': 'data'})):
        name = title.find('th').text
        if name == 'Planet Parameters':
            index = idx

    found_highlight = False
    for idx, text in enumerate(planet_props[index].findAll()):
        text = unidecode(str(text))

        # print()
        if 'th' in text:
            if 'tr' not in text:
                if 'class' not in text:
                    if 'span' not in text:
                        if '(' in text:
                            t = text[5:-6]
                            if 'sup' in t:
                                t = t.split('<')[0]
                            column_names_2.append(t)
                            # print(t)
                            
        if idx > 1:
            text_split = text.split('\n')
            # print(len(text_split))
            for t in text_split:
                # print(t)
                if 'td' in t:
                    if 'href' not in t:
                        t = t[4:-5]
                        if '+-' in t:
                            t = t.split('+-')[0].split(' ')[-1]
                            planets_2[p_idx-1].append(t)
                        elif 'lt' in t or 'gt' in t:
                            t = t.split(';')[-1]
                            planets_2[p_idx-1].append(t)
                        else:
                            t = t.split('span')
                            if len(t) == 1:
                                t = t[0].split(' ')[-1]
                                if 'null' not in t:
                                    if t.isdigit() or '.' in t:
                                        planets_2[p_idx-1].append(t)
                                    else:
                                        if len(t) == 1:
                                            planets_2.append([])
                                            # planets_2[p_idx].append(t)
                                            p_idx += 1
                                else:
                                    if len(t) == 1:
                                        planets_2[p_idx-1].append(t)
                                    elif 'null' in t:
                                        planets_2[p_idx-1].append(t)
                            else:
                                t = t[1].split('>')[1].split('<')[0]
                                # print(p_idx, end=', ')
                                planets_2[p_idx-1].append(t)
                    # else:
                    #     planets[p_idx-1].append('ref')

                # print(t)
    planets_2 = planets_2[::2]

    for i in range(len(planets)):
        planets[i].extend(planets_2[i])
    
    for i in column_names_2:
        column_names.append(i)
    # print(column_names)
   
    column_names = np.array(column_names)
    for c, col in enumerate(column_names):
        if col == 'Planet':
            column_names[c] = 'Name'
        if col == 'Period (days)':
            column_names[c] = 'n'
        if col == 'Semi-Major Axis (AU)':
            column_names[c] = 'a'
        if col == 'Inclination (deg)':
            column_names[c] = 'i'
        if col == 'Eccentricity':
            column_names[c] = 'e'
        if col == 'Longitude of Periastron (deg)':
            column_names[c] = 'pi'
        if col == 'Earth Mass':
            column_names[c] = 'Mass'
        if col == 'Jupiter Mass':
            column_names[c] = 'Mj'

    mass_idx = np.where(['Mass' in x for x in column_names])[0]
    column_names[mass_idx[-1]] = 'Mass_2'

    planets = np.array(planets)

    labels, n_data = np.unique(planets[:, 0], return_counts=True)
    start_idx = np.zeros_like(labels)
    for s, l in enumerate(labels):
        start_idx[s] = np.where(planets[:, 0] == l)[0][0]
        # print(np.where(planets == l)[0])
    start_idx = np.array(start_idx, dtype='int')

    planet_means = []

    for l in range(len(labels)):
        # print('________________', labels[l], '________________')
        planet_means.append([])
        planet_means[l].append(labels[l])
        for col in range(1, planets.shape[1]):
            col_data = planets[:, col][start_idx[l]:start_idx[l]+n_data[l]]
            # print(start_idx[l], n_data[l])
            planet_l_data = ma.masked_array(col_data, col_data == 'null').compressed()
            try:
                # print(column_names[col], ma.mean(ma.array(planet_l_data, dtype=float)))
                compressed_array = ma.array(planet_l_data, dtype=float)
                if len(compressed_array) > 0:
                    planet_means[l].append(ma.mean(compressed_array))
                else:
                    planet_means[l].append(np.nan)
            except:
                print('null')
    
    planet_means = np.array(planet_means)

    columns_to_ignore = ['Passage', 'Date', 'Mj', 'Radii', 'g/cm', 'K']
    idxs = []
    for word in columns_to_ignore:
        idx = np.where([word in x for x in column_names])[0]
        for i in idx:
            idxs.append(i)

    df = pd.DataFrame(columns=column_names, index=range(0, len(planet_means)))
    # print(df.shape, len(column_names))
    for row in range(len(planet_means)):
        for col in range(len(column_names)):
            df.ix[row, col] = planet_means[row, col]
            if planet_means[row, col] == 'null':
                df.ix[row, col] = np.nan
            if col in idxs:
                df.ix[row, col] = np.nan
    
    # df=df.dropna(axis=1, how='any')
    # df.to_csv('Test/'+obj_id+'.csv')
    # print(df)

    return df

def read_data(obj_id):

    obj_id_split = obj_id.split(' ')
    obj_id = obj_id_split[0]
    output_id = obj_id_split[0]
    for i in range(1, len(obj_id_split)):
        obj_id += '+'+obj_id_split[i]
        output_id += '_'+obj_id_split[i]

    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/ExoOverview/nph-ExoOverview?objname={}&type=&label&aliases&exo&iden&orb&ppar&tran&note&disc&ospar&ts&nalc&force=&dhxr1507830887922".format(obj_id)
    print('Data loading from\n'+url)
    response = requests.get(url)

    bs = BeautifulSoup(response.content, "html.parser")

    for idx, title in enumerate(bs.findAll('div', {'class': 'data'})):
        name = title.find('th').text
        if name == 'Planet Orbital Properties':
            index = idx
            # print(name, idx)

    planet_props = bs.findAll('div', {'class': 'data'})

    planets = []
    column_names = []
    p_idx = 0

    found_highlight = False
    for idx, text in enumerate(planet_props[index].findAll()):
        text = unidecode(str(text))

        found_highlight = 'class="overview_highlight"' in text
        # print(te)
        if 'th' in text:
            if 'class' not in text:
                if 'Reference' not in text:
                    # if 'td' not in text:
                    if 'href' not in text:
                        column_names.append(text[4:-5])
                # else:
                #     column_names.append(text.split('\n')[-1][4:-5])
                        # print(text[4:-5])
        if idx > 1:
            if found_highlight:
                text_split = text.split('\n')
                for t in text_split:
                    if 'tr' not in t:
                        if 'href' not in t:
                            t = t[4:-5]
                            if '+-' in t:
                                t = t.split('+-')[0].split(' ')[-1]
                                planets[p_idx-1].append(t)
                            elif 'lt' in t or 'gt' in t:
                                t = t.split(';')[-1]
                                planets[p_idx-1].append(t)
                            else:
                                t = t.split('span')
                                if len(t) == 1:
                                    t = t[0].split(' ')[-1]
                                    if 'null' not in t:
                                        if t.isdigit() or '.' in t:
                                            planets[p_idx-1].append(t)
                                        else:
                                            # print(t)
                                            planets.append([])
                                            planets[p_idx].append(t)
                                            p_idx += 1
                                    else:
                                        planets[p_idx-1].append(t)
                                    # print()
                                else:
                                    t = t[1].split('>')[1].split('<')[0]
                                    planets[p_idx-1].append(t)
                            # print(t)
            found_highlight=False

    p_idx = 0

    for idx, title in enumerate(bs.findAll('div', {'class': 'data'})):
        name = title.find('th').text
        if name == 'Planet Parameters':
            index = idx

    found_highlight = False
    for idx, text in enumerate(planet_props[index].findAll()):
        text = unidecode(str(text))

        found_highlight = 'class="overview_highlight"' in text
        # print()
        if 'th' in text:
            if 'tr' not in text:
                if 'class' not in text:
                    if 'span' not in text:
                        if '(' in text:
                            t = text[5:-6]
                            if 'sup' in t:
                                t = t.split('<')[0]
                            column_names.append(t)
                            # print(t)
                            
        if idx > 1:
            # print(found_highlight)
            if found_highlight:
                text_split = text.split('\n')
                # print(len(text_split))
                for t in text_split:
                    # print(t)
                    try:
                        if 'tr' not in t:
                            if 'href' not in t:
                                t = t[4:-5]
                                if '+-' in t:
                                    t = t.split('+-')[0].split(' ')[-1]
                                    planets[p_idx-1].append(t)
                                elif 'lt' in t or 'gt' in t:
                                    t = t.split(';')[-1]
                                    planets[p_idx-1].append(t)
                                else:
                                    t = t.split('span')
                                    if len(t) == 1:
                                        t = t[0].split(' ')[-1]
                                        if 'null' not in t:
                                            if t.isdigit() or '.' in t:
                                                planets[p_idx-1].append(t)
                                            else:
                                                p_idx += 1
                                        else:
                                            planets[p_idx-1].append(t)
                                    else:
                                        t = t[1].split('>')[1].split('<')[0]
                                        # print(p_idx, end=', ')
                                        planets[p_idx-1].append(t)
                            # else:
                            #     planets[p_idx-1].append('ref')
                    except:
                        print(t)
                    # print(t)
            found_highlight=False

    column_names = column_names[:]
    column_names = np.array(column_names)
    for c, col in enumerate(column_names):
        if col == 'Planet':
            column_names[c] = 'Name'
        if col == 'Period (days)':
            column_names[c] = 'n'
        if col == 'Semi-Major Axis (AU)':
            column_names[c] = 'a'
        if col == 'Inclination (deg)':
            column_names[c] = 'i'
        if col == 'Eccentricity':
            column_names[c] = 'e'
        if col == 'Longitude of Periastron (deg)':
            column_names[c] = 'pi'
        if col == 'Earth Mass':
            column_names[c] = 'Mass'
        if col == 'Jupiter Mass':
            column_names[c] = 'Mj'

    mass_idx = np.where(['Mass' in x for x in column_names])[0]
    column_names[mass_idx[-1]] = 'Mass_2'

    planets = np.array(planets)

    columns_to_ignore = ['Passage', 'Date', 'Mj', 'Radii', 'g/cm', 'K']
    idxs = []
    for word in columns_to_ignore:
        idx = np.where([word in x for x in column_names])[0]
        for i in idx:
            idxs.append(i)

    df = pd.DataFrame(columns=column_names, index=range(0, np.shape(planets)[0]))
    for row in range(np.shape(planets)[0]):
        for col in range(np.shape(planets)[1]):
            df.ix[row, col] = planets[row, col]
            if planets[row, col] == 'null':
                df.ix[row, col] = np.nan
            if col in idxs:
                # print(planets[row, col])
                df.ix[row, col] = np.nan
    
    # if np.sum(pd.isnull(df['pi'])) > 0:
    #     df['pi'] = search_pi(obj_id)

    # df=df.dropna(axis=1, how='any', thresh=0.8*len(planets))
    # df.to_csv('Exoplanets_data/'+output_id+'.csv', index=False)
    # print(df)

    for idx, title in enumerate(bs.findAll('div', {'class': 'data'})):
        name = title.find('th').text
        if name == 'Summary of Stellar Information':
            index = idx

    star_prop = []
    found_highlight = False
    for idx, text in enumerate(planet_props[index].findAll()):
        text = unidecode(str(text))

        found_highlight = 'Mass' in text
        if idx > 1:
            if found_highlight:
                text_split = text.split('\n')
                for t in text_split:
                    if 'tr' not in t:
                        if 'null' not in t:
                            if 'class' not in t:
                                t = t[4:-5].split('+-')
                                star_prop.append(float(t[0]))
                                # print(float(t[0]))
                                # print(len(t), t)
                found_highlight = False
    # print(star_prop)
    df1 = pd.DataFrame(columns=['star_mass', 'star_radius'], index=range(0, 1))
    df1.ix[0, 0] = star_prop[0]
    if len(star_prop) == 1:
        df1.ix[0, 1] = np.nan
    else:
        df1.ix[0, 1] = star_prop[1]
    # df1.to_csv('Exoplanets_data/'+output_id+'_star.csv', index=False)
    # for p in planets:
    #     print(p)
    #     print()

    return df, df1

def compare_data(df_highlighted, df_average, output_id):
    # print(df_highlighted.shape, df_average.shape)
    # s=0
    rows, cols = df_highlighted.shape

    mass_idx = df_highlighted.columns.get_loc("Mass")
    n_idx = df_highlighted.columns.get_loc("n")
    for row in range(rows):
        for col in range(cols):
            if pd.isnull(df_highlighted.ix[row, col]) and not pd.isnull(df_average.ix[row, col]):
                df_highlighted.ix[row, col] = df_average.ix[row, col]

            if col == mass_idx:
                if df_highlighted.ix[row, col] == 'nan':
                    df_highlighted.ix[row, col] = df_highlighted.ix[row, col+2]
                # print(df_highlighted.ix[row, col])
            if col == n_idx:
                df_highlighted.ix[row, col] = 2*np.pi/(float(df_highlighted.ix[row, col])/365)*180/np.pi

    # Period to n: 2pi/(period/365)*180/pi

    cols_to_keep = ['Name', 'n', 'a', 'e', 'i', 'pi', 'Mass']
    df_highlighted[cols_to_keep].to_csv('StarSystemData/'+output_id+'/'+'planets.csv', index=False)

if __name__ == '__main__':
    # obj_id = '55 Cnc'
    # obj_id = 'ups And'
    obj_id = 'HD 12661'
    # obj_id = '47 UMa'
    # obj_id = 'GJ 876'

    if len(sys.argv) > 1:
        obj_id = sys.argv[1]
        args = sys.argv[2:]
        for a in args:
            obj_id += ' '+a

    df_highlight, df_star = read_data(obj_id)
    df_mean = mean_data(obj_id)

    obj_id_split = obj_id.split(' ')
    output_id = obj_id_split[0]
    for i in range(1, len(obj_id_split)):
        output_id += '_'+obj_id_split[i]

    folder = glob.glob('StarSystemData/'+output_id)
    # os.system('mkdir '+obj_id)
    if len(folder) == 0:
        os.system('mkdir '+'StarSystemData/'+output_id)

    compare_data(df_highlight, df_mean, output_id)
    df_star.to_csv('StarSystemData/'+output_id+'/'+'star.csv', index=False)

    # print(df_highlight)
    # print(df_mean)
