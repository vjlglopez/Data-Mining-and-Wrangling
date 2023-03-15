from urllib.parse import urljoin
import urllib
import datetime


def summarize_items():
    return pd.read_html('https://jojie.accesslab.aim.edu:9095/'
                        'oasis/oi_item_summary.html', header=0)[0]


def item_values():
    return pd.read_html('https://jojie.accesslab.aim.edu:9095/'
                        'oasis/oi_d0700.html', header=1)[2]


def all_item_values():
    page = requests.get('https://jojie.accesslab.aim.edu:9095/'
                        'oasis/oi_item_summary.html')
    soup = bs4.BeautifulSoup(page.text)
    
    df_ans = pd.DataFrame()
    for item in soup.find_all('a', href=True):
        content = (
            pd.read_html('https://jojie.accesslab.aim.edu:9095/oasis/' +
                         item.get('href'), header=1)[2]
        )
        content.insert(loc=0, column='Item', value=item.text)
        df_ans = pd.concat([df_ans, content], ignore_index=True)

    return df_ans


def all_item_edits():
    path = 'https://jojie.accesslab.aim.edu:9095/oasis/'
    response = requests.get(urljoin(path, 'oe_index.html'))
    soup = bs4.BeautifulSoup(response.text, 'html.parser')

    content = soup.find_all('a', target='VIEW_WINDOW', href=True)

    df = pd.read_html(urljoin(path, content[0]['href']), header=0)[0]

    df['Edit Text'] = [
        pd.read_html(urljoin(path, i['href']),
                     header=0)[1].iloc[4, 1] for i in content[1:]
    ]
    df_ans = df.rename(columns={'Type': 'Edit Type'})
    return df_ans


def article_info(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)
    title_content = (
        soup.select_one('h1.post-single__title').get_text()
    )
    author_content = (
        soup.select_one('div.post-single__authors').get_text()
    )
    published_content = (
        soup.select_one('time.entry-date.published').get_text()
    )
    fin_dict = {
        'title': title_content,
        'author': author_content,
        'published': published_content
    }
    return fin_dict


def latest_news():
    path = 'https://jojie.accesslab.aim.edu:9095/rappler/'
    response = requests.get(path)
    soup = (
        bs4.BeautifulSoup(response.text, 'html.parser')
        .find(class_='latest-stories')
    )
    
    title_content = (
        soup.find(class_='post-card__title').text.strip()
    )
    category_content = (
        soup.find(class_='post-card__category').text.strip()
    )
    timestamp_content = (
        soup.find(class_="post-card__timeago")['datetime']
    )
    tc_datetime = datetime.datetime.fromisoformat(timestamp_content)
    
    fin_dict = {
        'title': title_content,
        'category': category_content,
        'timestamp': tc_datetime
    }
    return fin_dict


def get_category_posts(category):
    cat_clean = category.lower()

    url = 'https://jojie.accesslab.aim.edu:9095/rappler/'
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)

    categories = {}
    category_str = 'div.masthead-lower__container > ul > li > a'
    for i in soup.select(category_str):
        categories[i.text.lower()] = i['href']

    response_category = (
        requests.get(urljoin(url, categories[cat_clean]))
    )
    soup_cat = bs4.BeautifulSoup(response_category.content)
    
    filter_str = 'div.post-card__more-secondary-story h3 a'
    bs4_str = (
        soup_cat.select(categories[cat_clean][10:] + ' ' + filter_str)
    )
    
    fin_ans = []
    for i in bs4_str:
        fin_ans.append(i.text.strip())
    return fin_ans


def subsection_posts(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)

    title_content = soup.select('div.archive-article__content > h2 > a')
    timestamp_content = soup.select('div.archive-article__content > time')
    
    title = []
    for i in title_content:
        title.append(i.text.strip())
        
    timestamp = []
    for j in timestamp_content:
        timestamp.append(j.text.strip())

    df_ans = pd.DataFrame({'title': title,
                          'timestamp': timestamp})
    return df_ans


def subsection_authors(url):
    proxies = {'http': 'http://206.189.157.23'}
    response = requests.get(url, proxies=proxies)
    soup = (
        bs4.BeautifulSoup(response.text.encode('latin-1'), 'html.parser')
    )

    article_content = (
        soup.select('article .archive-article__content h2 a')
    )
    title = []
    author = []
    for article in article_content:
        response_new = requests.get(article['href'])
        soup_new = (
            bs4.BeautifulSoup(response_new.text)
            .find(class_='post-single__header')
        )
        title.append(soup_new.find('h1').text)
        author.append(soup_new.find(class_='post-single__authors').text)

    df_ans = pd.DataFrame({'title': title, 'author': author})
    return df_ans.sort_values('title')


def download_images(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)
    image_content = soup.select('#primary img')

    if not os.path.exists('images'):
        os.makedirs('images')
    else:
        pass

    for image in image_content:
        img_src = image['src']
        parsed = urllib.parse.urlparse(img_src)
        base = os.path.basename(parsed.path)
        
        with open(f'./images/{base}', 'wb') as f:
            response_new = requests.get(img_src)
            f.write(response_new.content)