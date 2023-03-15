import xml.etree.ElementTree as ET
from datetime import datetime


def establishment_info(establishment_id):
    response = (
        requests.get(f'https://jojie.accesslab.aim.edu:9095/'
                     f'rest/establishment/{establishment_id}',).json()
    )
    del response['establishment_id']
    return response


def check_in(establishment_id, year, month, day, hour, minute, seconds):
    check_in_time = (
        datetime(year, month, day, hour, minute, seconds).isoformat()
    )
    url = 'https://jojie.accesslab.aim.edu:9095/rest/visit/check-in'
    json = (
        {"establishment_id": establishment_id,
         "checkin_ts": check_in_time}
    )
    fin_ans = requests.post(url, json=json).json()
    return fin_ans


def visits(establishment_id, start_date, end_date):
    date_visits = (
        {i.strftime('%Y-%m-%d'): '' 
         for i in pd.date_range(start_date, end_date)}
    )
    for j in date_visits.keys():
        date_visits[j] = (
            requests.get(f'https://jojie.accesslab.aim.edu:9095/'
                         f'rest/establishment/{establishment_id}/visits',
                         params={'date': j}).json()
        )['visits']
    return date_visits


def pageprops(title):
    response = (
        requests.get(f'https://en.wikipedia.org/w/api.php',
                     params={'action': 'query', 
                             'titles': 'Philippines',
                             'prop': 'pageprops',
                             'format': 'json'}).json()
    )
    fin_ans = response['query']['pages'].values()
    lst_fin_ans = list(fin_ans)[0]['pageprops']
    return lst_fin_ans


def contributors(revid):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
         'action': 'query',
         'prop': 'revisions',
         'revids': revid,
         'format': 'xml'
    }
    root = ET.fromstring(requests.get(url=url, params=params).text)
    pageid = root.find(".//page").attrib['pageid']
    params = {
        'action': 'query',
        'prop': 'revisions',
        'pageids': pageid,
        'rvprop': 'userid|user',
        'rvstart': '2022-10-24T00:00:00Z',
        'rvlimit': 'max',
        'format': 'json'
    }
    
    lst_contrib = []
    while True:
        response = requests.get(url=url, params=params).json()
        lst_contrib.extend(
            response['query']['pages'][str(pageid)]['revisions']
        )
        if 'continue' in response:
            params.update(response['continue'])
        else:
            break
    
    df = (
        pd.DataFrame(lst_contrib)
        [['userid', 'user']].dropna().drop_duplicates()
    )
    cols = ['userid', 'name']
    df.columns = cols
    df['userid'] = df['userid'].astype(int)
    return (
        df.drop(index=df[df['userid']==0].index)
        .sort_values(by='userid', ignore_index=True)
    )


def revisions(title):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
         'action': 'query',
         'prop': 'revisions',
         'titles': title,
         'format': 'xml'
    }

    root = ET.fromstring(requests.get(url=url, params=params).text)
    pageid = root.find(".//page").attrib['pageid']
    params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': title,
            'rvprop': 'ids|user|timestamp|sha1',
            'rvstart': '2007-01-01T00:00:00Z',
            'rvlimit': 'max',
            'format': 'json'
    }
    
    lst_revs = []
    while True:
        response = requests.get(url=url, params=params).json()
        lst_revs.extend(
            response['query']['pages'][str(pageid)]['revisions'])
        if 'continue' in response:
            params.update(response['continue'])
        else:
            break
    
    cols = ['revid', 'user', 'timestamp', 'sha1']
    df = pd.DataFrame(lst_revs)[cols]
    fin_ans = df.sort_values(by='revid', ignore_index=True)
    return fin_ans


def nearby_drugs(api_key):
    coordinates = ['14.552665405264284, 121.01868822115196']
    url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
    params = {
        'location': coordinates,
        'keyword': ['drug'],
        'rankby': ['distance'],
        'key': api_key
    }
    
    closest_drg = []
    while len(closest_drg) < 50:
        response = requests.get(url=url, params=params).json()
        np_token = response['next_page_token']
        params['next_page_token'] = np_token
        closest_drg.extend(response['results'])
    cols = ['name','vicinity']
    df = pd.DataFrame(closest_drg[:50])[cols]
    return df


def account_info(username, bearer_token):
    response = (
        requests.get(f'https://api.twitter.com/2/users/'
                     f'by/username/{username}',
                     headers={'Authorization': f'Bearer {bearer_token}'},
                     params={"user.fields": "location,created_at"}
                    ).json()['data']
    )
    return response


def tweets_2021(user_id, bearer_token):
    url = f'https://api.twitter.com/2/users/{user_id}/tweets'
    params={
        'tweet.fields': 'id,created_at,text',
        'start_time': '2021-01-01T00:00:00.000Z',
        'end_time': '2022-01-01T00:00:00.000Z',
        'max_results': 100
    }
    
    lst_tweets = []
    while True:
        response = (
            requests.get(url=url,
                         headers={'Authorization': f'Bearer {bearer_token}'},
                         params=params).json()
        )
        lst_tweets.extend(response['data'])
        if 'next_token' in response['meta']:
            params['pagination_token'] = response['meta']['next_token']
        else:
            break
    cols = ['id', 'created_at', 'text']
    df = pd.DataFrame(lst_tweets)[cols]
    fin_ans = df.sort_values(by='created_at', ignore_index=True)
    return fin_ans


def crawl_page():
    url = (
        'https://jojie.accesslab.aim.edu:9095/'
        'messages?fixed-token=gniparcs-bew'
    )
    response = requests.get(url)
    while "continue" in response.json():
        response = (
            requests.get(url + f"&continue={response.json()['continue']}")
        )
    return response.json()['message']