"""Create dataset from raw data."""
import copy
import math
import os
import random
import unicodedata

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def create_dataset(
    config: DictConfig
) -> None:
    """Create dataset

    Args:
        config: Experiment settings.
    """
    data = load_data(
        config.raw_data.dir,
        config.raw_data.filename,
    )
    for ind in range(1, len(data) + 1):
        data.loc[ind, 'name_1'] = delete_extra(
            data.loc[ind, 'name_1'],
            config.replaced_symbols,
            config.allowed_symbols,
        )
        data.loc[ind, 'name_2'] = delete_extra(
            data.loc[ind, 'name_2'],
            config.replaced_symbols,
            config.allowed_symbols,
        )
    data = data[(data.name_1 != '') & (data.name_2 != '')]
    data_duplicate = data[data.is_duplicate == 1]
    data_not_duplicate = data[data.is_duplicate == 0]
    dict_comp = create_data_dict(
        data_duplicate,
        data_not_duplicate,
    )
    dict_comp_old = copy.deepcopy(dict_comp)
    if config.generate_data:
        dict_comp = generate_data(dict_comp)
    write_dataset(
        dict_comp,
        dict_comp_old,
        config.data.dir,
        config.data.filename,
    )


def load_data(
    data_dir: str,
    data_filename: str,
) -> pd.core.frame.DataFrame:
    """Load raw data.

    Args:
        data_dir: Data file directory.
        data_filename: Data file name.

    Returns:
        DataFrame with raw data.
    """
    data = pd.read_csv(
        os.path.join(
            data_dir,
            data_filename,
        ),
        index_col='pair_id',
    )
    return data


def delete_extra(
    name: str,
    replaced_symbols: str,
    allowed_symbols: str,
) -> str:
    """Removing extra characters and checking for the correctness of the name.

    Args:
        name: Company name.
        replaced_symbols: Replacement characters.
        allowed_symbols: Allowed characters.

    Returns:
        Company name after change.
    """
    for symb_rep in replaced_symbols:
        name = name.replace(symb_rep, '')
    name = deaccent(name)
    for symb_name in name:
        if symb_name not in allowed_symbols:
            return ''
    name_arr = name.split()
    return (' '.join(name_arr))


def deaccent(
    name: str,
) -> str:
    """Convert accented chars to their non-accented equivalent.

    Args:
        name: Company name.

    Returns:
        Company name after сonvert.
    """
    NON_NFKD_MAP = {
        '\u0181': 'B', '\u1d81': 'd', '\u1d85': 'l',
        '\u1d89': 'r', '\u028b': 'v', '\u019d': 'N',
        '\u1d8d': 'x', '\u1d83': 'g', '\u0191': 'F',
        '\u0220': 'N', '\u01a5': 'p', '\u0224': 'Z',
        '\u0126': 'H', '\u01ad': 't', '\ua74a': 'O',
        '\u01b5': 'Z', '\u0234': 'l', '\u023c': 'c',
        '\u0240': 'z', '\u0142': 'l', '\u024c': 'R',
        '\u0244': '', '\u2c60': 'L', '\u0248': 'J',
        '\ua752': 'P', '\ua756': 'Q', '\ua75a': 'R',
        '\ua75e': 'V', '\u0260': 'g', '\u1d72': 'r',
        '\u01e5': 'g', '\u2c64': 'R', '\u0166': 'T',
        '\u0268': 'i', '\u2c66': 't', '\u2c74': 'v',
        '\u026c': 'l', '\u1d6e': 'f', '\u1d87': 'n',
        '\u1d76': 'z', '\u2c78': 'e', '\u027c': 'r',
        '\u1eff': 'y', '\ua741': 'k', '\u1d96': 'i',
        '\u0182': 'B', '\u1d86': 'm', '\u0288': 't',
        '\u018a': 'D', '\u1d8e': 'z', '\u019a': 'l',
        '\u0111': 'd', '\u0290': 'z', '\u0192': 'f',
        '\u019e': 'n', '\u1d88': 'p', '\u02a0': 'q',
        '\u01ae': 'T', '\u01b2': 'V', '\u024b': 'q',
        '\u01b6': 'z', '\u023b': 'C', '\u023f': 's',
        '\u0141': 'L', '\u0243': 'B', '\ua74d': 'o',
        '\ua745': 'k', '\u0247': 'e', '\ua749': 'l',
        '\u024f': 'y', '\ua751': 'p', '\u0253': 'b',
        '\ua755': 'p', '\u0257': 'd', '\u0291': 'z',
        '\ua759': 'q', '\xd8': 'O', '\u2c63': 'P',
        '\u2c67': 'H', '\u026b': 'l', '\u1d6d': 'd',
        '\u1d71': 'p', '\u0273': 'n', '\u1d75': 't',
        '\u1d91': 'd', '\xf8': 'o', '\u2c7e': 'S',
        '\u1d7d': 'p', '\u2c7f': 'Z', '\u0183': 'b',
        '\u0187': 'C', '\u1d80': 'b', '\u0110': 'D',
        '\u0289': '', '\u018b': 'D', '\u1d8f': 'a',
        '\u0193': 'G', '\u1d82': 'f', '\u0197': 'I',
        '\u029d': 'j', '\u019f': 'O', '\ua744': 'K',
        '\u2c6c': 'z', '\u01ab': 't', '\u01b3': 'Y',
        '\u0236': 't', '\u023a': 'A', '\u0246': 'E',
        '\u023e': 'T', '\ua740': 'K', '\u1d8a': 's',
        '\ua748': 'L', '\ua74c': 'O', '\u024e': 'Y',
        '\ua750': 'P', '\ua754': 'P', '\u1d70': 'n',
        '\u0256': 'd', '\ua758': 'Q', '\u2c62': 'L',
        '\u0266': 'h', '\u2c73': 'w', '\u0272': 'n',
        '\u2c6a': 'k', '\u1d6c': 'b', '\u2c6e': 'M',
        '\u1d92': 'e', '\u1d74': 's', '\u2c7a': 'o',
        '\u2c6b': 'Z', '\u027e': 'r', '\u1d8c': 'v',
        '\u0180': 'b', '\u0282': 's', '\u1d84': 'k',
        '\u0188': 'c', '\u018c': 'd', '\u0221': 'd',
        '\ua742': 'K', '\u1d99': '', '\u0198': 'K',
        '\u2c71': 'v', '\u0225': 'z', '\u01a4': 'P',
        '\u0127': 'h', '\u01ac': 'T', '\ua753': 'p',
        '\u0235': 'n', '\u01b4': 'y', '\u2c72': 'W',
        '\u023d': 'L', '\ua743': 'k', '\u0255': 'c',
        '\u0249': 'j', '\ua74b': 'o', '\u024d': 'r',
        '\ua757': 'q', '\u2c68': 'h', '\ua75b': 'r',
        '\ua75f': 'v', '\u2c61': 'l', '\u027d': 'r',
        '\u2c65': 'a', '\u01e4': 'G', '\u0167': 't',
        '\u2c69': 'K', '\u026d': 'l', '\u1efe': 'Y',
        '\u1d6f': 'm', '\u0271': 'm', '\u1d73': 'r',
        '\u0199': 'k',
    }
    return ''.join(
        NON_NFKD_MAP[c] if c in NON_NFKD_MAP else c for part in
        unicodedata.normalize('NFKD', name) for c in part if
        unicodedata.category(part) != 'Mn'
    )


def create_data_dict(
    data_duplicate: pd.core.frame.DataFrame,
    data_not_duplicate: pd.core.frame.DataFrame,
) -> dict:
    """Create dictionary with all companies.

    Args:
        data_duplicate: DataFrame with company duplicates.
        data_not_duplicate: DataFrame with different pairs of companies.

    Returns:
        Dictionary with all companies.
    """
    dict_comp = {}
    for ind in data_duplicate.index:
        name_1 = data_duplicate.loc[ind, 'name_1']
        name_2 = data_duplicate.loc[ind, 'name_2']
        key = min(
            get_key(dict_comp, name_1),
            get_key(dict_comp, name_2),
        )
        if key in dict_comp.keys():
            dict_comp[key] |= set([name_1, name_2])
        else:
            dict_comp[key] = set([name_1, name_2])
    for ind in data_not_duplicate.index:
        name_1 = data_not_duplicate.loc[ind, 'name_1']
        key_1 = get_key(dict_comp, name_1)
        if key_1 in dict_comp.keys():
            dict_comp[key_1] |= set([name_1])
        else:
            dict_comp[key_1] = set([name_1])

        name_2 = data_not_duplicate.loc[ind, 'name_2']
        key_2 = get_key(dict_comp, name_2)
        if key_2 in dict_comp.keys():
            dict_comp[key_2] |= set([name_2])
        else:
            dict_comp[key_2] = set([name_2])
    return dict_comp


def get_key(
    dictionary: dict,
    value: str
) -> int:
    """Getting a key in a dictionary by value.

    Args:
        dictionary: Dictionary.
        value: Value from dictionary.

    Returns:
        Key from dictionary.
    """
    key = 0
    for key, set in dictionary.items():
            if value in set:
                return key
    return key + 1


def generate_data(
    dict_comp: dict,
) -> dict:
    """Сompany name generation.

    Args:
        dict_comp: Dictionary with company names.

    Returns:
        Dictionary with new company names.
    """
    list_of_countries = [
        'AW', 'Aruba', 'AF', 'Afghanistan', 'AO', 'Angola', 'AI', 'Anguilla',
        'AX', 'Åland Islands', 'AL', 'Albania', 'AD',
        'Andorra', 'AE', 'United Arab Emirates', 'AR', 'Argentina', 'AM',
        'Armenia', 'AS', 'American Samoa', 'AQ',
        'Antarctica', 'TF', 'French Southern Territories', 'AG',
        'Antigua and Barbuda', 'AU', 'Australia', 'AT', 'Austria',
        'AZ', 'Azerbaijan', 'BI', 'Burundi', 'BE', 'Belgium', 'BJ', 'Benin',
        'BQ', 'Bonaire, Sint Eustatius and Saba',
        'BF', 'Burkina Faso', 'BD', 'Bangladesh', 'BG', 'Bulgaria', 'BH',
        'Bahrain', 'BS', 'Bahamas', 'BA',
        'Bosnia and Herzegovina', 'BL', 'Saint Barthélemy', 'BY', 'Belarus',
        'BZ', 'Belize', 'BM', 'Bermuda', 'BO',
        'Bolivia, Plurinational State of', 'BR', 'Brazil', 'BB', 'Barbados',
        'BN', 'Brunei Darussalam', 'BT', 'Bhutan',
        'BV', 'Bouvet Island', 'BW', 'Botswana', 'CF',
        'Central African Republic', 'CA', 'Canada', 'CC',
        'Cocos (Keeling) Islands', 'CH', 'Switzerland', 'CL',
        'Chile', 'CN', 'China', 'CI', "Côte d'Ivoire", 'CM',
        'Cameroon', 'CD', 'Congo, The Democratic Republic of the',
        'CG', 'Congo', 'CK', 'Cook Islands', 'CO', 'Colombia',
        'KM', 'Comoros', 'CV', 'Cabo Verde', 'CR', 'Costa Rica', 'CU',
        'Cuba', 'CW', 'Curaçao', 'CX', 'Christmas Island',
        'KY', 'Cayman Islands', 'CY', 'Cyprus', 'CZ', 'Czechia', 'DE',
        'Germany', 'DJ', 'Djibouti', 'DM', 'Dominica', 'DK',
        'Denmark', 'DO', 'Dominican Republic', 'DZ', 'Algeria', 'EC',
        'Ecuador', 'EG', 'Egypt', 'ER', 'Eritrea', 'EH',
        'Western Sahara', 'ES', 'Spain', 'EE', 'Estonia', 'ET', 'Ethiopia',
        'FI', 'Finland', 'FJ', 'Fiji', 'FK',
        'Falkland Islands (Malvinas)', 'FR', 'France', 'FO', 'Faroe Islands',
        'FM', 'Micronesia, Federated States of', 'Guernsey',
        'GA', 'Gabon', 'GB', 'United Kingdom', 'GE', 'Georgia', 'GG',
        'GH', 'Ghana', 'GI', 'Gibraltar', 'GN',
        'Guinea', 'GP', 'Guadeloupe', 'GM', 'Gambia', 'GW', 'Guinea-Bissau',
        'GQ', 'Equatorial Guinea', 'GR', 'Greece',
        'GD', 'Grenada', 'GL', 'Greenland', 'GT', 'Guatemala', 'GF',
        'French Guiana', 'GU', 'Guam', 'GY', 'Guyana', 'HK',
        'Hong Kong', 'HM', 'Heard Island and McDonald Islands', 'HN',
        'Honduras', 'HR', 'Croatia', 'HT', 'Haiti', 'HU',
        'Hungary', 'ID', 'Indonesia', 'IM', 'Isle of Man', 'IN', 'India',
        'IO', 'British Indian Ocean Territory', 'IE',
        'Ireland', 'IR', 'Iran, Islamic Republic of', 'IQ', 'Iraq', 'IS',
        'Iceland', 'IL', 'Israel', 'IT', 'Italy', 'JM',
        'Jamaica', 'JE', 'Jersey', 'JO', 'Jordan', 'JP', 'Japan', 'KZ',
        'Kazakhstan', 'KE', 'Kenya', 'KG', 'Kyrgyzstan',
        'KH', 'Cambodia', 'KI', 'Kiribati', 'KN', 'Saint Kitts and Nevis',
        'KR', 'Korea, Republic of', 'KW', 'Kuwait',
        'LA', "Lao People's Democratic Republic", 'LB', 'Lebanon', 'LR',
        'Liberia', 'LY', 'Libya', 'LC', 'Saint Lucia',
        'LI', 'Liechtenstein', 'LK', 'Sri Lanka', 'LS', 'Lesotho', 'LT',
        'Lithuania', 'LU', 'Luxembourg', 'LV', 'Latvia',
        'MO', 'Macao', 'MF', 'Saint Martin (French part)', 'MA', 'Morocco',
        'MC', 'Monaco', 'MD', 'Moldova, Republic of',
        'MG', 'Madagascar', 'MV', 'Maldives', 'MX', 'Mexico', 'MH',
        'Marshall Islands', 'MK', 'North Macedonia', 'ML',
        'Mali', 'MT', 'Malta', 'MM', 'Myanmar', 'ME', 'Montenegro',
        'MN', 'Mongolia', 'MP', 'Northern Mariana Islands',
        'MZ', 'Mozambique', 'MR', 'Mauritania', 'MS', 'Montserrat',
        'MQ', 'Martinique', 'MU', 'Mauritius', 'MW', 'Malawi',
        'MY', 'Malaysia', 'YT', 'Mayotte', 'NA', 'Namibia', 'NC',
        'New Caledonia', 'NE', 'Niger', 'NF', 'Norfolk Island',
        'NG', 'Nigeria', 'NI', 'Nicaragua', 'NU', 'Niue', 'NL',
        'Netherlands', 'NO', 'Norway', 'NP', 'Nepal', 'NR',
        'Nauru', 'NZ', 'New Zealand', 'OM', 'Oman', 'PK', 'Pakistan', 'PA',
        'Panama', 'PN', 'Pitcairn', 'PE', 'Peru', 'PH',
        'Philippines', 'PW', 'Palau', 'PG', 'Papua New Guinea', 'PL',
        'Poland', 'PR', 'Puerto Rico', 'KP',
        "Korea, Democratic People's Republic of", 'PT', 'Portugal', 'PY',
        'Paraguay', 'PS', 'Palestine, State of', 'PF',
        'French Polynesia', 'QA', 'Qatar', 'RE', 'Réunion', 'RO', 'Romania',
        'RU', 'Russian Federation', 'RW', 'Rwanda',
        'SA', 'Saudi Arabia', 'SD', 'Sudan', 'SN', 'Senegal', 'SG',
        'Singapore', 'GS',
        'South Georgia and the South Sandwich Islands', 'SH',
        'Saint Helena, Ascension and Tristan da Cunha', 'SJ',
        'Svalbard and Jan Mayen', 'SB', 'Solomon Islands', 'SL',
        'Sierra Leone', 'SV', 'El Salvador', 'SM', 'San Marino',
        'SO', 'Somalia', 'PM', 'Saint Pierre and Miquelon', 'RS', 'Serbia',
        'SS', 'South Sudan', 'ST',
        'Sao Tome and Principe', 'SR', 'Suriname', 'SK', 'Slovakia', 'SI',
        'Slovenia', 'SE', 'Sweden', 'SZ', 'Eswatini',
        'SX', 'Sint Maarten (Dutch part)', 'SC', 'Seychelles', 'SY',
        'Syrian Arab Republic', 'TC',
        'Turks and Caicos Islands', 'TD', 'Chad', 'TG', 'Togo', 'TH',
        'Thailand', 'TJ', 'Tajikistan', 'TK', 'Tokelau',
        'TM', 'Turkmenistan', 'TL', 'Timor-Leste', 'TO', 'Tonga', 'TT',
        'Trinidad and Tobago', 'TN', 'Tunisia', 'TR',
        'Turkey', 'TV', 'Tuvalu', 'TW', 'Taiwan, Province of China', 'TZ',
        'Tanzania, United Republic of', 'UG', 'Uganda',
        'UA', 'Ukraine', 'UM', 'United States Minor Outlying Islands', 'UY',
        'Uruguay', 'US', 'United States', 'UZ',
        'Uzbekistan', 'VA', 'Holy See (Vatican City State)', 'VC',
        'Saint Vincent and the Grenadines', 'VE',
        'Venezuela, Bolivarian Republic of', 'VG', 'Virgin Islands, British',
        'VI', 'Virgin Islands, U.S.', 'VN',
        'Viet Nam', 'VU', 'Vanuatu', 'WF', 'Wallis and Futuna', 'WS', 'Samoa',
        'YE', 'Yemen', 'ZA', 'South Africa', 'ZM',
        'Zambia', 'ZW', 'Zimbabwe',
    ]
    terms_by_type = {
        'Corporation': [
            'company', 'incorporated', 'corporation', 'corp.', 'corp', 'inc',
            '& co.', '& co', 'inc.', 's.p.a.', 'n.v.', 'a.g.', 'ag', 'nuf',
            's.a.', 's.f.', 'oao', 'co.', 'co',
        ],
        'General Partnership': [
            'soc.col.', 'stg', 'd.n.o.', 'ltda.', 'v.o.s.', 'a spol.',
            've\xc5\x99. obch. spol.', 'kgaa', 'o.e.', 's.f.', 's.n.c.',
            's.a.p.a.', 'j.t.d.', 'v.o.f.', 'sp.j.', 'og', 'sd', ' i/s',
            'ay', 'snc', 'oe', 'bt.', 's.s.', 'mb',
            'ans', 'da', 'o.d.', 'hb', 'pt',
        ],
        'Joint Stock / Unlimited': [
            'unltd', 'ultd', 'sal', 'unlimited', 'saog', 'saoc', 'aj',
            'yoaj', 'oaj', 'akc. spol.', 'a.s.',
        ],
        'Joint Venture': ['esv', 'gie', 'kv.', 'qk'],
        'Limited': [
            'pty. ltd.', 'pty ltd', 'ltd', 'l.t.d.', 'bvba', 'd.o.o.',
            'g.m.b.h', 'kft.', 'kht.', 'zrt.', 'ehf.', 's.a.r.l.',
            'b.v.', 'tapui', 'ltda', 'gmbh', 'd.o.o.e.l.', 's. de r.l.',
            'sp. z.o.o.', 'sp. z o.o.', 'spółka z o.o.',
            's.r.l.', 's.l.', 's.l.n.e.', 'ood', 'oy', 'rt.',
            'teo', 'uab', 'scs', 'sprl', 'limited', 'bhd.',
            'lda.', 'tov', 'pp', 'sdn. bhd.', 'sdn bhd', 'as',
        ],
        'Limited Liability Company': [
            'pllc', 'llc', 'l.l.c.', 'plc.', 'plc', 'hf.', 'oyj',
            'a.e.', 'nyrt.', 'p.l.c.', 'sh.a.', 's.a.', 's.r.l.',
            'd.d.', 'srl.', 'srl', 'aat', '3at', 'sa', 'aps',
            's.r.o.', 'spol. s r.o.', 's.m.b.a.', 'smba', 'sarl', 'nv',
            'a/s', 'p/s', 'sae', 'sasu', 'eurl', 'ae', 'cpt', 'as', 'ab',
            'vat', 'zat', 'mchj', 'a.d.', 'asa', 'ooo', 'dat',
        ],
        'Limited Liability Limited Partnership': ['lllp', 'l.l.l.p.'],
        'Limited Liability Partnership': [
            'llp', 'l.l.p.', 'sp.p.', 's.c.a.', 's.c.s.',
        ],
        'Limited Partnership': [
            'gmbh & co. kg', 'lp', 'l.p.', 's.c.s.', 's.a.s.', 's. en c.',
            's.c.p.a', 'comm.v', 'k.d.', 'k.d.a.', 's. en c.', 'e.e.',
            'c.v.', 's.k.a.', 'sp.k.', 's.cra.', 'ky', 'scs', 'kg', 'kd',
            'kda', 'ks', 'kb', 'kt', 'k/s', 'ee', 'secs',
        ],
        'Mutual Fund': ['sicav'],
        'No Liability': ['nl'],
        'Non-Profit': ['vzw', 'ses.', 'gte.'],
        'Private Company': ['private', 'pte', 'xk'],
        'Professional Corporation': ['p.c.', 'vof', 'snc'],
        'Professional Limited Liability Company': [
            'pllc', 'p.l.l.c.',
        ],
        'Sole Proprietorship': [
            'e.u.', 's.p.', 't:mi', 'tmi', 'e.v.', 'e.c.', 'et', 'obrt',
            'fie', 'ij', 'fop', 'xt',
        ],
    }
    terms_by_country = {
        'Albania': ['sh.a.', 'sh.p.k.'],
        'Argentina': [
            's.a.', 's.r.l.', 's.c.p.a', 'scpa', 's.c.e i.', 's.e.', 's.g.r',
            'soc.col.',
        ],
        'Australia': ['nl', 'pty. ltd.', 'pty ltd'],
        'Austria': ['e.u.', 'stg', 'gesbr', 'a.g.', 'ag', 'og', 'kg'],
        'Belarus': ['aat', '3at'],
        'Belgium': [
            'esv', 'vzw', 'vof', 'snc', 'comm.v', 'scs', 'bvba',

            'cvoa', 'sca', 'sep', 'gie', 'sprl', 'cvba',
        ],
        'Bosnia / Herzegovina': [
            'd.d.', 'a.d.', 'd.n.o.', 'd.o.o.', 'k.v.', 's.p.',
        ],
        'Brazil': [
            'ltda', 's.a.', 'pllc', 'ad', 'adsitz',
            'ead', 'et', 'kd', 'kda', 'sd',
        ],
        'Bulgaria': ['ad', 'adsitz', 'ead', 'et', 'kd', 'kda', 'sd'],
        'Cambodia': [
            'gp', 'sm pte ltd.', 'pte ltd.', 'plc ltd.', 'peec', 'sp',
        ],
        'Canada': ['gp', 'lp', 'sp'],
        'Chile': [
            'eirl', 's.a.', 'sgr', 's.g.r.', 'ltda',
            's.p.a.', 'sa', 's. en c.', 'ltda.',
        ],
        'Columbia': ['s.a.', 'e.u.', 's.a.s.', 'suc. de descendants', 'sca'],
        'Croatia': ['d.d.', 'd.o.o.', 'obrt'],
        'Czech Republic': [
            'a.s.', 'akc. spol.', 's.r.o.', 'spol. s r.o.',
            'v.o.s.', 've\xc5\x99. obch. spol.',
            'a spol.', 'k.s.', 'kom. spol.', 'kom. spol.',
        ],
        'Denmark': ['i/s', 'a/s', 'k/s', 'p/s', 'amba',
                    's.m.b.a.', 'g/s', 'a.m.b.a.', 'fmba', 'f.m.b.a.', 'smba',
                    ],
        'Dominican Republic': ['c. por a.', 'cxa', 's.a.', 's.a.s.',
                               'sas', 'srl.', 'srl', 'eirl.', 'sa',
                               ],
        'Ecuador': ['s.a.', 'c.a.', 'sa', 'ep'],
        'Egypt': ['sae'],
        'Estonia': ['fie'],
        'Finland': [
            't:mi', 'tmi', 'as oy', 'as.oy',
            'ay', 'ky', 'oy', 'oyj', 'ok',
        ],
        'France': [
            'sicav', 'sarl', 'sogepa', 'ei', 'eurl', 'sasu', 'snc',
            'scs', 'sca', 'scop', 'sem', 'sas', 'fcp', 'gie', 'sep',
        ],
        'Germany': ['gmbh & co. kg', 'e.g.', 'e.v.', 'gbr', 'ohg', 'partg',
                    'kgaa', 'gmbh', 'g.m.b.h.', 'ag', 'mbh & co. kg',
                    ],
        'Greece': ['a.e.', 'ae', 'e.e.', 'ee', 'epe', 'e.p.e.', 'mepe',
                   'oe', 'ovee', 'o.v.e.e.', 'm.e.p.e.', 'o.e.',
                   ],
        'Guatemala': ['s.a.', 'sa'],
        'Haiti': ['sa'],
        'Hong Kong': ['ltd', 'unltd', 'ultd', 'limited'],
        'Hungary': ['e.v.', 'e.c.', 'bt.', 'kft.', 'kht.',
                    'ev', 'ec', 'rt.', 'kkt.', 'k.v.', 'zrt.', 'nyrt',
                    ],
        'Iceland': ['ehf.', 'hf.', 'ohf.', 's.f.', 'ses.'],
        'India': ['pvt. ltd.', 'ltd.', 'psu', 'pse'],
        'Indonesia': ['ud', 'fa', 'pt'],
        'Ireland': ['cpt', 'teo'],
        'Israel': ['b.m.', 'bm', 'ltd', 'limited'],
        'Italy': [
            's.n.c.', 's.a.s.', 's.p.a.', 's.a.p.a.',
            's.r.l.', 's.c.r.l.', 's.s.',
        ],
        'Latvia': ['as', 'sia', 'ik', 'ps', 'ks'],
        'Lebanon': ['sal'],
        'Lithuania': ['uab', 'ab', 'ij', 'mb'],
        'Luxemborg': ['s.a.', 's.a.r.l.', 'secs'],
        'Macedonia': [
            'd.o.o.', 'd.o.o.e.l', 'k.d.a.', 'j.t.d.', 'a.d.', 'k.d.',
        ],
        'Malaysia': ['bhd.', 'sdn. bhd.'],
        'Mexico': ['s.a.', 's. de. r.l.', 's. en c.', 's.a.b.', 's.a.p.i.'],
        'Mongolia': ['xk', 'xxk'],
        'Netherlands': ['v.o.f.', 'c.v.', 'b.v.', 'n.v.'],
        'New Zealand': ['tapui', 'ltd', 'limited'],
        'Nigeria': ['gte.', 'plc', 'ltd.', 'ultd.'],
        'Norway': [
            'asa', 'as', 'ans', 'ba', 'bl', 'da', 'etat', 'fkf',
            'ks', 'nuf', 'rhf', 'sf', 'hf', 'iks', 'kf',
        ],
        'Oman': ['saog', 'saoc'],
        'Pakistan': ['ltd.', 'pvt. ltd.', 'ltd', 'limited'],
        'Peru': ['sa', 's.a.', 's.a.a.'],
        'Philippines': ['coop.', 'corp.', 'corp', 'ent.', 'inc.',
                        'ltd.', 'inc', 'llc', 'l.l.c.',
                        ],
        'Poland': [
            'p.p.', 's.k.a.', 'sp.j.', 'sp.k.', 'sp.p.',
            'sp. z.o.o.', 's.c.', 's.a.',
        ],
        'Portugal': ['lda.', 'crl', 's.a.', 's.f.', 'sgps'],
        'Romania': ['s.c.a.', 's.c.s.', 's.n.c.', 's.r.l.', 'o.n.g.', 's.a.'],
        'Russia': ['ooo', 'oao', 'zao', 'pao', 'oao', 'ooo'],
        'Serbia': ['d.o.o.', 'a.d.', 'k.d.', 'o.d.'],
        'Singapore': [
            'bhd', 'pte ltd', 'sdn bhd', 'llp',
            'l.l.p.', 'ltd.', 'pte', 'pte. ltd.',
        ],
        'Slovenia': ['d.d.', 'd.o.o.', 'd.n.o.', 'k.d.', 's.p.'],
        'Slovakia': [
            'a.s.', 'akc. spol.', 's.r.o.', 'spol. s r.o.',
            'k.s.', 'kom. spol.', 'v.o.s.', 'a spol.',
        ],
        'Spain': ['s.a.', 's.a.d.', 's.l.', 's.l.l.', 's.l.n.e.',
                  'sal', 'sccl', 's.c.', 's.cra', 's.coop',
                  ],
        'Sweden': ['ab', 'hb', 'kb'],
        'Switzerland': ['ab', 'sa', 'gmbh', 'g.m.b.h.', 'sarl', 'sagl'],
        'Turkey': ['koop.'],
        'Ukraine': [
            'dat', 'fop', 'kt', 'pt', 'tdv', 'tov', 'pp', 'vat', 'zat', 'at',
        ],
        'United Kingdom': [
            'plc.', 'plc', 'cic', 'cio', 'l.l.p.', 'llp', 'l.p.', 'lp', 'ltd.',
            'lp', 'ltd.', 'ltd', 'limited',
        ],
        'United States of America': [
            'llc', 'inc.', 'corporation', 'incorporated', 'company',
            'limited', 'corp.', 'inc.', 'inc', 'llp', 'l.l.p.', 'pllc',
            '& company', 'inc', 'inc.', 'corp.', 'corp', 'ltd.', 'ltd',
            'co', 'lp', '& co.', '& co', 'co.', 'and company',
        ],
        'Uzbekistan': [
            'mchj', 'qmj', 'aj', 'oaj', 'yoaj', 'xk', 'xt', 'ok', 'uk', 'qk',
        ],
    }
    for index in dict_comp.keys():
        # каждый класс до 5 представителей
        if len(dict_comp[index]) >= 5:
            continue

        elif len(dict_comp[index]) > 1:
            # поиск медианы представителей в классе
            # определение кол-ва повторений каждого слова
            lens_name = []
            dict_word = {}
            for name in dict_comp[index]:
                lens_name.append((name.count(' ') + 1))
                for word in name.split():
                    if word.lower() in dict_word.keys():
                        dict_word[word.lower()] += 1
                    else:
                        dict_word[word.lower()] = 1
            median = np.median(lens_name)
            max_count = max(dict_word.values())
            # паттер из самых часто встречающихся слов в классе
            pattern = ''
            for key, val in dict_word.items():
                if val == max_count:
                    pattern += key + ' '
            if 2 * len(pattern.split()) < median:
                max_count -= 1
                pattern = ''
                for key, val in dict_word.items():
                    if val >= max_count:
                        pattern += key + ' '
        # для классов с одним представителем
        else:
            pattern = ''
            median = (name.count(' ') + 1) / 2
            name = list(dict_comp[index])[0]
            for word in name.split()[:math.ceil(len(name.split()) / 2)]:
                pattern += word + ' '
        pattern = pattern.title()

        # добавление в словарь паттерна
        dict_comp[index] |= set([pattern.rstrip()])
        while len(dict_comp[index]) < 5:
            # удаление последних слов паттерна с некоторой вероятностью
            if len(pattern.split()) >= 3:
                if (random.random() >= 0.5):
                    new_name = ''
                    for i in range(len(pattern.split()) - 1):
                        if random.random() >= 0.9:
                            new_name += pattern.split()[i].upper() + ' '
                        else:
                            new_name += pattern.split()[i] + ' '

                elif (len(pattern.split()) >= 4) and (random.random() >= 0.4):
                    new_name = ''
                    count_del_word = math.trunc(median / 2 * random.random())
                    for i in range(len(pattern.split()) - count_del_word):
                        if random.random() >= 0.9:
                            new_name += pattern.split()[i].upper() + ' '
                        else:
                            new_name += pattern.split()[i] + ' '
                else:
                    if random.random() >= 0.9:
                        new_name = pattern.upper()
                    else:
                        new_name = pattern
            else:
                if random.random() >= 0.9:
                    new_name = pattern.upper()
                else:
                    new_name = pattern

            new_name = new_name.rstrip()

            # показатели порога вероятности добавления нового слова
            # меняется от 1 до 2
            threshold_second_add = 1
            threshold_country_add = 1

            # добавление к новому имени слов из словаря terms_by_type
            if random.random() >= 0.5:
                type_add = random.choice(list(terms_by_type.keys()))
                new_name += ' ' + random.choice(terms_by_type[type_add])
                threshold_second_add += 0.2
                threshold_country_add += 0.1
                if random.random() >= 0.2:
                    type_add = random.choice(list(terms_by_type.keys()))
                    new_name += ' ' + random.choice(terms_by_type[type_add])
                    threshold_second_add += 0.5

            # добавление к новому имени слов из словаря terms_by_country
            if random.random() > 0.5 * threshold_second_add:
                country_add = random.choice(list(terms_by_country.keys()))
                new_name += ' ' + random.choice(terms_by_country[country_add])
                threshold_country_add += 0.7

            # добавление к новому имени слов из списка стран
            if random.random() > 0.5 * threshold_country_add:
                new_name += ' ' + random.choice(list_of_countries)

            # добавление нового имени в словарь
            if delete_extra(new_name) == '':
                continue
            dict_comp[index] |= set([deaccent(new_name)])
    return dict_comp


def write_dataset(
    dict_comp: dict,
    dict_comp_old: dict,
    data_dir: str,
    data_filename: str,
) -> None:
    """Write dataset to file.

    Args:
        dict_comp: Dictionary with company names.
        dict_comp_old: Dictionary with company names without generated names.
        data_dir: Data file directory.
        data_filename: Data file name.
    """
    with open(
        os.path.join(
            data_dir,
            data_filename,
        ),
        'w',
        encoding="utf-8",
    ) as file_handler:
        for key, value in dict_comp.items():
            for i in range(len(value)):
                if list(value)[i] in dict_comp_old[key]:
                    file_handler.write(
                        '{0};{1};0\n'.format(str(list(value)[i]), str(key)),
                    )
                else:
                    file_handler.write(
                        '{0};{1};1\n'.format(str(list(value)[i]), str(key)),
                    )


if __name__ == '__main__':
    create_dataset()
