# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental molecule geometries from the
 Computational Chemistry Comparison and Benchmark DataBase (CCCDBD)

Ref:
https://cccbdb.nist.gov/expgeom1x.asp
"""

import numpy as np
import pandas as pd
import requests
from absl import logging
from bs4 import BeautifulSoup


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


def query_cccbdb(formula: str, calculation: str = "expgeom") -> BeautifulSoup:
  data = {'formula': formula, 'submit1': 'Submit'}

  url1 = f'https://cccbdb.nist.gov/{calculation}1x.asp'
  url2 = f'https://cccbdb.nist.gov/{calculation}2x.asp'

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
  return soup


def extract_geometry(soup: BeautifulSoup) -> str:
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


def query_geometry_from_cccbdb(formula: str) -> str:
  soup = query_cccbdb(formula, "expgeom")
  return extract_geometry(soup)


def parse_result_table_to_df(soup: BeautifulSoup) -> pd.DataFrame:
  table = soup.find("table", attrs={"id": "table2"})

  headers = [header.text for header in table.find('tr').find_all('th')]

  # Get table rows
  rows = table.find_all('tr')[1:]

  # Extract row data
  data = []
  index_l1s = []
  index_l2s = []
  index_l1 = index_l2 = None
  for row in rows:
    indexes = row.find_all('th')
    if len(indexes) == 2:
      index_l1 = indexes[0].text
      index_l2 = indexes[1].text
    else:
      index_l2 = indexes[0].text
    index_l1s.append(index_l1)
    index_l2s.append(index_l2)
    columns = []
    for col in row.find_all('td'):
      try:
        columns.append(float(col.text))
      except Exception as e:
        print(e)
        columns.append(np.nan)
    data.append(columns)

  # Convert to pandas DataFrame
  df = pd.DataFrame(data, columns=headers)
  df.index = pd.MultiIndex.from_arrays(
    [index_l1s, index_l2s], names=('Level1', 'Level2')
  )

  return df


def query_calc_from_cccbdb(formula: str) -> pd.DataFrame:
  soup = query_cccbdb(formula, "energy")
  return parse_result_table_to_df(soup)
