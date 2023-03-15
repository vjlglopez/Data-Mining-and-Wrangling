def list_recipes(recipe_xml):
    root = ET.fromstring(recipe_xml)
    return [child.text for child in root.findall('ingredient')]


def catalog_sizes(catalog_xml):
    root = ET.fromstring(catalog_xml)
    fin_ans = (
        sorted(list(set(
            [child.get('description') for child in root.findall('.//size')]
        )))
    )
    return fin_ans


def read_fantasy(books_xml):
    df = pd.read_xml(books_xml)
    df.drop(df[df['genre'] != 'Fantasy'].index, inplace = True)
    return df


def dct2str(dct):
    jsonStr = json.dumps(dct)
    return jsonStr


def dct2file(dct):
    with open("dct.json", "w") as fp:
        return json.dump(dct,fp)
    

def count_journals():
    df = pd.read_json(
        '/mnt/data/public/covid19-lake/'
        'alleninstitute/CORD19/json/metadata/'
        'part-00000-81803174-7752-4489-8eeb-081318af9653-c000.json',
        lines=True)
    df_new = df['journal'].value_counts().reset_index()
    df_fin = df_new.sort_values(['journal', 'index'],
                                ascending = [False, True])
    df_fin['final'] = df_fin[['index', 'journal']].apply(tuple, axis=1)
    return df_fin['final'].to_list()


def business_labels():
    df = pd.read_json('/mnt/data/public/yelp/challenge12/'
                      'yelp_dataset/yelp_academic_dataset_photo.json',
                      lines=True)
    df_fin = df.groupby(['business_id'])['label'].apply(set)
    return df_fin


def get_businesses():
    path = open('/mnt/data/public/yelp/'
                'challenge12/yelp_dataset/'
                'yelp_academic_dataset_business.json',
                'r')
    cols = []
    for lines in path.readlines():
        cols.append(json.loads(lines))
    fin = (
        pd.json_normalize(cols)
        .set_index('business_id')
        .head(10_000)
          )
    return fin


def pop_ncr():
    df = pd.read_excel('/mnt/data/public/'
                       'census/2020/NCR.xlsx',
                       sheet_name='NCR by barangay',
                       skiprows=4)
    df.drop(axis=1, columns=[df.columns[0], df.columns[1]], inplace=True)
    df.rename(
        columns={'and Barangay': 'Province, City, Municipality, and Barangay',
                 'Population': 'Total Population'},
        inplace=True
    )
    df.dropna(inplace=True)
    return df


def dump_airbnb_beds():
    df = pd.read_csv('/mnt/data/public/insideairbnb/'
                     'data.insideairbnb.com/united-kingdom/'
                     'england/london/2015-04-06/data/listings.csv.gz',
                     compression='gzip',
                     usecols=['bed_type',
                              'host_location',
                              'price'])
    with pd.ExcelWriter('airbnb.xlsx') as writer:
        for bed_type in sorted(df.bed_type.unique()):
            df_new = df[df.bed_type == bed_type].copy()
            df_new.drop(columns='bed_type', inplace=True)
            df_new.to_excel(writer, sheet_name=bed_type, index=False)
            
            
def age_counts():
    df1 = pd.read_excel('/mnt/data/public/census/2015/'
                        '_PHILIPPINES_Statistical Tables.xls',
                        'T2', skiprows=2,
                        usecols=['Single-Year Age', 'Both Sexes',
                                 'Male', 'Female'],
                        index_col='Single-Year Age')
    df2 = pd.read_excel('/mnt/data/public/census/2015/'
                        '_PHILIPPINES_Statistical Tables.xls',
                        'T3', skiprows=2,
                        usecols=['Single-Year Age', 'Both Sexes',
                                 'Male', 'Female'],
                        index_col='Single-Year Age')
    df1_new = df1.loc['Under  1':'80 years and over']
    df2_new = df2.loc['Under  1':'80 years and over']
    df_fin = df1_new.merge(df2_new, left_index=True,
                           right_index=True,
                           suffixes=(' (Total)', ' (Household)'))
    return df_fin
