"""Experimental molecule geometries from the
 Computational Chemistry Comparison and Benchmark DataBase (CCCDBD)

Ref:
https://cccbdb.nist.gov/expgeom1x.asp
"""

import requests
from absl import logging
from bs4 import BeautifulSoup

# some example geometry
h2_geometry = """
H 0.0000 0.0000 0.0000;
H 0.0000 0.0000 0.7414;
"""

h2o_geometry = """
O 0.0000 0.0000 0.1173;
H 0.0000 0.7572 -0.4692;
H 0.0000 -0.7572 -0.4692;
"""

o2_geometry = """
O 0.0000 0.0000 0.0000;
O 0.0000 0.0000 1.2075;
"""

co2_geometry = """
C 0.0000 0.0000 0.0000
O 0.0000 0.0000 1.1621
O 0.0000 0.0000 -1.1621
"""


def headers(referer):
  return {
    'Host':
      'cccbdb.nist.gov',
    'Connection':
      'keep-alive',
    'Content-Length':
      '26',
    'Pragma':
      'no-cache',
    'Cache-Control':
      'no-cache',
    'Origin':
      'http://cccbdb.nist.gov',
    'Upgrade-Insecure-Requests':
      '1',
    'User-Agent':
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
      (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Content-Type':
      'application/x-www-form-urlencoded',
    'Accept':
      'text/html,application/xhtml+xml,application/xml;q=0.9,\
      image/webp,*/*;q=0.8',
    'Referer':
      referer,
    'Accept-Encoding':
      'gzip, deflate',
    'Accept-Language':
      'en-CA,en-GB;q=0.8,en-US;q=0.6,en;q=0.4',
  }


URLS = {'form': 'https://cccbdb.nist.gov/getformx.asp'}


def query_geometry_from_cccbdb(
  formula: str, calculation: str = "expgeom"
) -> str:
  data = {'formula': formula, 'submit1': 'Submit'}

  url1 = 'https://cccbdb.nist.gov/%s1x.asp' % calculation
  url2 = 'https://cccbdb.nist.gov/%s2x.asp' % calculation

  logging.info('**** Posting formula')

  # request initial url
  session = requests.Session()
  res = session.post(
    URLS['form'], data=data, headers=headers(url1), allow_redirects=False
  )

  logging.info('**** Fetching data')

  # follow the redirect
  if res.status_code == 302:
    res2 = session.get(url2)

  soup = BeautifulSoup(res2.content, 'html.parser')
  table = soup.find('table', attrs={'class': 'border'})

  d = table.text.strip().split('\n')[5:]
  assert len(d) % 4 == 0

  # NOTE: -1 to remove counter after atom symbol
  geometry = "".join(
    [
      f"{d[i].strip()[:-1]}  "
      f"{d[i+1].strip()} {d[i+2].strip()} {d[i+3].strip()}\n"
      for i in range(0, len(d), 4)
    ]
  )
  return geometry
