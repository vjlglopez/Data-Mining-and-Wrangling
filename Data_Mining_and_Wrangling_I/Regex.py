def is_date(test_str):
    return bool(re.match(r'(?<!\d)(\d{2}|\d{4})(?!\d)([-/])\d{2}\2'
                         '(?<!\d)(\d{2}|\d{4})(?!\d)',
                         test_str))


def find_data(text):
    return re.findall(r'(data set\b|dataset\b)', text)


def find_lamb(text):
    return re.findall(r'\blittle.*?lamb\b', text)


def repeat_alternate(text):
    return re.sub(r"(\w+'?\w*\s)(\w+'?\w*)", r'\g<1>\g<1>\g<2>', text)


def whats_on_the_bus(text):
    items = re.findall(r'(The )(\w*)', text)
    return list(set([i[1] for i in items]))


def to_list(text):
    return re.split(r',|\+|and', text)


def march_product(text):
    res = re.findall(r'(\d*) by (\d*)', text)
    return [(int(x) * int(y)) for x, y in res]


def get_big(items):
    matches = (
        re.findall(r'([A-Z][0-9]|[0-9][A-Z]|[A-Z]{2})\s(Big\b.*)', items)
    )
    return [match[1] for match in matches]


def find_chris(names):
    return re.findall(r'(\w*[Cc]hris\w*)\s[^BM]', names)


def get_client(server_log):
    return (
        re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - -'
                   r'\[(.*)\] ".*" (\d{3})', server_log)
    )