"""Create a Discourse topic for every book page that has no Discussions link.

Each new page gets one topic, shared by all framework tabs, titled
"<page title> (<chapter title>)" in the English Version category; its first
post is the page's d2l.ai URL. The script then inserts an untabbed
`[Discussions](https://d2l.discourse.group/t/<id>)` line before the page's
`<!-- slides -->` marker (or at the end of the file).

Needs an admin API key in ~/.config/d2l/discourse.env (DISCOURSE_API_KEY=...).

    python3 tools/create_forum_topics.py          # dry run: list pages
    python3 tools/create_forum_topics.py --create # create topics, insert links
    python3 tools/create_forum_topics.py --create --files a.md b.md
                                                  # only these pages
"""
import json
import os
import re
import sys
import time
import urllib.request

FORUM = 'https://d2l.discourse.group'
CATEGORY = 5  # English Version


def pages_without_link():
    toc = open('_quarto.yml').read()
    for qmd in re.findall(r'(chapter_[^\s]+\.qmd)', toc):
        path = qmd.replace('.qmd', '.md')
        if path.endswith('/index.md'):
            continue
        text = open(path).read()
        if re.search(r'd2l\.discourse\.group/t/\d+', text):
            continue
        title = re.search(r'^# (.+)$', text, re.M).group(1).strip()
        index = open(os.path.join(os.path.dirname(path), 'index.md')).read()
        chapter = re.search(r'^# (.+)$', index, re.M).group(1).strip()
        yield path, f'{clean(title)} ({clean(chapter)})'


def clean(title):
    return title.replace('`', '').replace('--', '–')


def create_topic(key, title, url):
    req = urllib.request.Request(
        f'{FORUM}/posts.json', method='POST',
        data=json.dumps(dict(title=title, raw=url, category=CATEGORY)).encode(),
        headers={'Api-Key': key, 'Api-Username': 'system',
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)['topic_id']


def insert_link(path, topic_id):
    text = open(path).read()
    link = f'[Discussions]({FORUM}/t/{topic_id})\n'
    i = text.find('\n<!-- slides -->')
    if i >= 0:
        text = text[:i].rstrip('\n') + '\n\n' + link + text[i:]
    else:
        text = text.rstrip('\n') + '\n\n' + link
    open(path, 'w').write(text)


def main():
    todo = list(pages_without_link())
    if '--files' in sys.argv:
        wanted = set(a for a in sys.argv[sys.argv.index('--files') + 1:]
                     if not a.startswith('--'))
        unknown = wanted - {path for path, _ in todo}
        if unknown:
            sys.exit(f'not in the book or already linked: {sorted(unknown)}')
        todo = [(path, title) for path, title in todo if path in wanted]
    if '--create' not in sys.argv:
        for path, title in todo:
            print(f'{path:70s} {title}')
        print(f'{len(todo)} pages without a Discussions link')
        return
    env = os.path.expanduser('~/.config/d2l/discourse.env')
    key = dict(line.strip().split('=', 1) for line in open(env)
               if '=' in line)['DISCOURSE_API_KEY']
    for path, title in todo:
        url = 'https://d2l.ai/' + path.replace('.md', '.html')
        topic_id = create_topic(key, title, url)
        insert_link(path, topic_id)
        print(topic_id, path)
        time.sleep(1.5)


if __name__ == '__main__':
    main()
