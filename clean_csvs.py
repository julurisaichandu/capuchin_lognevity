import pandas as pd
import os

def convert_and_standardize_units(df):
    # ——————————————————————————————————————
    # 1) Collapse synonyms into base‐unit labels
    # ——————————————————————————————————————
    df['result'] = pd.to_numeric(df['result'], errors='coerce')

    unit_synonyms = {

# variants of “10³/µL” or “per µL”
        '10X3':       '/uL',
        '10^3/ul':    '/uL',
        'x10^3/µl':   '/uL',
        'THOUS/CMM':  '/uL',
        'THDS/CMM':   '/uL',

        # variants of “10⁶/µL”
        '10X6':       '10^6/uL',
        '10^6/ul':    '10^6/uL',
        'x10^6/µl':   '10^6/uL',
        'MILL/CMM':   '10^6/uL',

        # variants of “per mL”
        'X 1000/ML':  '10^3/mL',
        'X 1 MIL/ML': '10^6/mL',

        # stray femtoliter alias
        'U3':         'fL',


        # Millions per µL
        'M/μL':       '10^6/uL',  'M/uL':    '10^6/uL',
        '10^6/μL':    '10^6/uL',  'x 1000000/ul': '10^6/uL',
        '10^6/mL':    '10^6/mL',  # handle mL separately below

        # Thousands per µL or raw /µL
        'K/μL':       '/uL',      'K/uL':    '/uL',
        '10^3/μL':    '/uL',      '103/μL':  '/uL',
        'x 1000/ul':  '/uL',      '10^3/mL': '10^3/mL',
        '/uL':        '/uL',      'µL':      '/uL',
        'uL':         '/uL',

        # Billions per L
        '10^9/L':     '10^9/L',   'x10^9/L': '10^9/L',

        # Mass/volume
        'µg/dL':      'ug/dL',    'ug/dl':   'ug/dL',
        'MG/DL':      'mg/dL',    'mg/dl':   'mg/dL',
        'g/dl':       'g/dL',     'GM%':     'g/dL',
        'G/DL':       'g/dL',

        # Other lab units
        'μmol/L':     'umol/L',   'MICROMOLES/L': 'umol/L',
        'mmol/l':     'mmol/L',   'mEq/L':        'mmol/L',
        'FL':         'fL',       'fl':           'fL',
        'PG':         'pg',       'pg':           'pg',
        'UUG':        'pg',

        'IU/L':       'U/L',      'U-3':          'U/L',
        'RATIO':      '',         'Ratio':        ''
    }
    df['units'] = df['units'].replace(unit_synonyms)

    # ——————————————————————————————————————
    # 2) Numeric rescaling + final unit label
    # ——————————————————————————————————————
    conversions = {
        # raw cells per µL → cells per 10^3/µL
        '/uL':    (1/1_000, '10^3/uL'),

        # 10^3 per mL → cells per µL
        '10^3/mL': (1/1_000, '/uL'),

        # 10^6 per mL → 10^3 per µL
        '10^6/mL': (1/1_000, '10^3/uL'),

        # 10^9 per L → 10^3 per µL (no numeric change)
        '10^9/L': (1.0,     '10^3/uL'),

        # mass conversions
        'ug/dL':  (1/1_000, 'mg/dL'),  # µg → mg
        # mg/dL, g/dL etc can be left as‐is:
        'mg/dL':  (1.0,     'mg/dL'),
        'g/dL':   (1.0,     'g/dL'),
        'umol/L': (1.0,     'umol/L'),
        'mmol/L': (1.0,     'mmol/L'),
        'U/L':    (1.0,     'U/L'),
        'fL':     (1.0,     'fL'),
        'pg':     (1.0,     'pg'),
        '':       (1.0,     ''),       # for ratios
    }

    for base_unit, (factor, new_unit) in conversions.items():
        mask = df['units'] == base_unit
        if mask.any():
            df.loc[mask, 'result'] *= factor
            df.loc[mask, 'units']  = new_unit

    return df


def clean_capuchin_data(file_path):
    """
    Reads a raw capuchin lab data CSV and performs a three-step cleaning process:
    1. Standardizes test names (case-insensitively).
    2. Filters out unwanted/qualitative tests.
    3. Standardizes units for all tests.
    
    Args:
        file_path (str): The path to the input CSV file.

    Returns:
        pandas.DataFrame: A cleaned and standardized DataFrame.
    """
    # --- Step 0: Read and Basic Prep ---
    print(f"--- Processing {os.path.basename(file_path)} ---")
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.columns = df.columns.str.strip()
    df['test'] = df['test'].str.strip()
    df['units'] = df['units'].astype(str).str.strip() # Ensure units column is string type for processing

    # --- Step 1: Standardize Test Names (Now Case-Insensitive) ---
    # This dictionary maps all known variations (in lowercase) to a single standard name.
    test_name_mapping_lower = {
        'ast (sgot)': 'AST', 'ast': 'AST',
        'alt (sgpt)': 'ALT', 'alt': 'ALT',
        'alk phosphatase': 'ALP', 'alkaline phosphatase': 'ALP',
        'urea nitrogen': 'BUN', 'bun': 'BUN',
        'creatinine, serum': 'Creatinine', 'creatinine': 'Creatinine', 'crea': 'Creatinine',
        'glucose, serum': 'Glucose', 'glucose': 'Glucose', 'glu': 'Glucose',
        'ca': 'Calcium', 'calcium': 'Calcium',
        'phos': 'Phosphorus', 'phosphorus': 'Phosphorus',
        'tp': 'Total Protein', 'total protein': 'Total Protein',
        'alb': 'Albumin', 'albumin': 'Albumin',
        'glob': 'Globulin', 'globulin, total': 'Globulin', 'globulin': 'Globulin',
        'tbil': 'Total Bilirubin', 'total bilirubin': 'Total Bilirubin', 'bilirubin, total': 'Total Bilirubin',
        'tco2 (bicarbonate)': 'TCO2', 'tco2': 'TCO2',
        'red blood cell count': 'RBC', 'rbc': 'RBC',
        'white blood cell count': 'WBC', 'wbc': 'WBC',
        'hgb': 'Hemoglobin', 'hemoglobin': 'Hemoglobin',
        'hct': 'Hematocrit', 'hematocrit': 'Hematocrit',
        'platelets': 'Platelets', 'platelet': 'Platelets', 'platelet count': 'Platelets', 'auto platelet': 'Platelets', 'plt': 'Platelets',
        '% neutrophil': '% Neutrophils', '% neutrophils': '% Neutrophils',
        '% lymphocyte': '% Lymphocytes', '% lymphocytes': '% Lymphocytes', 'lymf (%)': '% Lymphocytes',
        '% monocyte': '% Monocytes', '% monocytes': '% Monocytes', 'mono (%)': '% Monocytes',
        '% eosinophil': '% Eosinophils', '% eosinophils': '% Eosinophils',
        '% basophil': '% Basophils', '% basophils': '% Basophils',
        'neutrophil': 'Neutrophils (Absolute)', 'neutrophils': 'Neutrophils (Absolute)', 'neutrophils (absolute)': 'Neutrophils (Absolute)',
        'lymphocyte': 'Lymphocytes (Absolute)', 'lymphocytes': 'Lymphocytes (Absolute)', 'lymphocytes (absolute)': 'Lymphocytes (Absolute)',
        'monocyte': 'Monocytes (Absolute)', 'monocytes': 'Monocytes (Absolute)', 'monocytes (absolute)': 'Monocytes (Absolute)',
        'eosinophil': 'Eosinophils (Absolute)', 'eosinophils': 'Eosinophils (Absolute)', 'eosinophils (absolute)': 'Eosinophils (Absolute)',
        'basophil': 'Basophils (Absolute)', 'basophils': 'Basophils (Absolute)', 'basophils (absolute)': 'Basophils (Absolute)',
        'na': 'Sodium', 'sodium': 'Sodium',
        'k': 'Potassium', 'potassium': 'Potassium',
        'cl': 'Chloride', 'chloride': 'Chloride',
        'chol': 'Cholesterol', 'cholesterol': 'Cholesterol',
        'amyl': 'Amylase', 'amylase': 'Amylase',
        'creatine kinase': 'Creatine Kinase', 'cpk': 'Creatine Kinase',
        'idexx sdma': 'SDMA', 'sdma': 'SDMA',
        'bun/creatinine ratio': 'BUN:Creatinine Ratio', 'bun: creatinine ratio': 'BUN:Creatinine Ratio',
        'a/g ratio': 'A:G Ratio', 'alb/glob ratio': 'A:G Ratio', 'albumin: globulin ratio': 'A:G Ratio', 'albumin/globulin ratio': 'A:G Ratio',
        'na/k ratio': 'Na:K Ratio', 'na: k ratio': 'Na:K Ratio',
        'bilirubin - conjugated': 'Bilirubin (Conjugated)', 'bilirubin (conjugated)': 'Bilirubin (Conjugated)',
        'bilirubin - unconjugated': 'Bilirubin (Unconjugated)', 'bilirubin (unconjugated)': 'Bilirubin (Unconjugated)'
    }
    # Apply the mapping case-insensitively
    df['test'] = df['test'].str.lower().map(test_name_mapping_lower).fillna(df['test'])

    # --- Step 2: Filter for a common set of quantitative tests ---
    tests_to_keep = [
        'RBC', 'WBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Platelets',
        '% Neutrophils', '% Lymphocytes', '% Monocytes', '% Eosinophils', '% Basophils',
        'Neutrophils (Absolute)', 'Lymphocytes (Absolute)', 'Monocytes (Absolute)',
        'Eosinophils (Absolute)', 'Basophils (Absolute)', 'Bands (Absolute)',
        'Glucose', 'BUN', 'Creatinine', 'BUN:Creatinine Ratio', 'Phosphorus', 'Calcium',
        'Sodium', 'Potassium', 'Na:K Ratio', 'Chloride', 'TCO2', 'Anion Gap',
        'Total Protein', 'Albumin', 'Globulin', 'A:G Ratio',
        'ALT', 'AST', 'ALP', 'GGT', 'Total Bilirubin', 'Bilirubin (Conjugated)', 'Bilirubin (Unconjugated)',
        'Cholesterol', 'Amylase', 'Lipase', 'Creatine Kinase', 'SDMA', 'Fructosamine',
        'Magnesium', 'Triglycerides'
    ]
    df = df[df['test'].isin(tests_to_keep)].copy()
    print(f"  Step 1 & 2: Standardized test names and filtered to {len(df['test'].unique())} common tests.")

    # --- Step 3: Standardize Units ---
    convert_and_standardize_units(df)

    print("  Step 3: Standardized and filled missing units.")

    # Final: coerce results numeric, drop NA, sort
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    before = len(df)
    df.dropna(subset=['result'], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} non-numeric rows.")
    df.sort_values(['date','test'], inplace=True)
    return df.reset_index(drop=True)

# def normalize_units(df):
#     # simply re-use the same logic
#     return convert_and_standardize_units(df)

def filter_common_tests(dfs):
    common = set.intersection(*(set(df['test'].unique()) for df in dfs.values()))
    print(f"✅ Retaining {len(common)} tests common to all files.")
    return { name: df[df['test'].isin(common)].copy() for name, df in dfs.items() }

"""
capuchin_pipeline.py

1. clean_capuchin_data(file): your existing clean-up→filter→unit-synonym logic
2. determine_target_units(dfs): picks the most-common unit per test across all cleaned DataFrames
3. normalize_units(df, target_units): applies the per-test conversions and drops unconvertible rows

Finally writes out *_cleaned.csv and *_normalized.csv for each input file.
"""

import os
import glob
import pandas as pd
import warnings
from collections import Counter, defaultdict


# ─── 2) Determine target units ──────────────────────────────────────────────────

def determine_target_units(dfs):
    """
    Given a dict of {name: cleaned_df}, count unit frequencies for each test
    and return a dict { test_name: most_common_unit }.
    """
    unit_counts = defaultdict(Counter)
    for df in dfs.values():
        for test, grp in df.groupby('test'):
            unit_counts[test].update(grp['units'].dropna().tolist())

    # pick the highest‐count unit for each test
    return {
        test: counts.most_common(1)[0][0]
        for test, counts in unit_counts.items()
    }


# ─── 3) Normalize (convert) to target units ────────────────────────────────────

# your per‐test conversion rules
ABSOLUTE_TESTS = {
    'Basophils (Absolute)',
    'Eosinophils (Absolute)',
    'Lymphocytes (Absolute)',
    'Monocytes (Absolute)',
    'Neutrophils (Absolute)',
}

def normalize_units(df, target_unit):
    """
    Converts every row of df['result'] into target_unit[df.test]:
    - % → absolute via WBC on same date
    - 10^3/uL → 10^6/uL for RBC
    - % → g/dL for MCHC
    Drops rows it can’t convert (e.g. missing WBC).
    Returns the normalized DataFrame.
    """
    rows = []
    for _, row in df.iterrows():
        test, unit, val, date = row['test'], row['units'], row['result'], row['date']
        tgt = target_unit.get(test, unit)

        # no change
        if unit == tgt:
            rows.append((test, date, val, unit))
            continue

        # differential % → absolute
        if test in ABSOLUTE_TESTS and unit == '%' and tgt == '10^3/µL':
            wbc = df.loc[(df.test=='WBC (Absolute)') & (df.date==date), 'result']
            if wbc.empty:
                continue  # drop row
            val = val/100 * wbc.iloc[0]
            rows.append((test, date, val, tgt))
            continue

        # RBC 10^3/uL → 10^6/uL
        if test=='RBC' and unit=='10^3/µL' and tgt=='10^6/µL':
            rows.append((test, date, val/1000, tgt))
            continue

        # MCHC % → g/dL
        if test=='MCHC' and unit=='%' and tgt=='g/dL':
            rows.append((test, date, val, tgt))
            continue

        # all other cases: warn, relabel, keep
        warnings.warn(f"No conversion rule for {test} {unit}→{tgt}, relabel only.")
        rows.append((test, date, val, tgt))

    out = pd.DataFrame(rows, columns=['test','date','result','units'])
    # reattach any other columns if you want, or merge back on index
    return out

# --- 4) Check for any remaining unit mismatches -------------------------------

def check_unnormalized_units(files, target_map):
    """
    Returns a dict of {test: {filename: set(units)}} where any unit != target_map[test].
    """
    issues = defaultdict(lambda: defaultdict(set))
    for fp in files:
        df = pd.read_csv(fp)
        for test, grp in df.groupby('test'):
            units = set(grp['units'].dropna().unique())
            tgt = target_map.get(test)
            # include tests where any observed unit differs from target
            if tgt and any(u != tgt for u in units):
                issues[test][os.path.basename(fp)] = units
    return issues

# --- 5) Drop rows for missing-unit tests ---------------------------------------

def find_tests_with_missing_units(files):
    missing = set()
    for fp in files:
        df = pd.read_csv(fp)
        tc = next(c for c in df.columns if 'test' in c.lower())
        uc = next(c for c in df.columns if 'unit' in c.lower())
        df[uc] = df[uc].replace(r'^\s*$', pd.NA, regex=True)
        missing.update(df[df[uc].isna()][tc].unique())
    return missing


def drop_missing_tests(df, tests_to_drop):
    return df[~df['test'].isin(tests_to_drop)].copy()


# --- Main Script Execution ---
if __name__ == '__main__':
    raw_files=['Allie one deceased.csv','Annie one deceased.csv','Bambi one living.csv','daisy one.csv','davey one.csv']

    cleaned = {}
    for path in raw_files:
        if path.endswith('_cleaned.csv') or path.endswith('_normalized.csv') or path.endswith('_final.csv'):
            continue
        dfc = clean_capuchin_data(path)
        outc = path.replace('.csv','_cleaned.csv')
        dfc.to_csv(outc, index=False)
        cleaned[outc] = dfc
        print(f"Wrote cleaned: {outc}")

    target_map = determine_target_units(cleaned)
    print("\nTarget units per test:")
    for test, unit in target_map.items(): print(f"  {test}: {unit}")

    normalized_files = []
    for name, dfc in cleaned.items():
        dfn = normalize_units(dfc, target_map)
        outn = name.replace('_cleaned.csv','_normalized.csv')
        dfn.to_csv(outn, index=False)
        normalized_files.append(outn)
        print(f"Wrote normalized: {outn}")

    # New: check for any unit mismatches post-normalization
    issues = check_unnormalized_units(normalized_files, target_map)
    if issues:
        print("\nTests with unexpected units after normalization:")
        for test, file_units in issues.items():
            print(f"- {test}:")
            for fname, units in file_units.items():
                print(f"    {fname}: {', '.join(units)}")
    else:
        print("\nAll units match target units after normalization.")

    # Drop tests with missing units
    to_drop = find_tests_with_missing_units(normalized_files)
    print("\nDropping tests with missing units:")
    for t in sorted(to_drop): print(f"  - {t}")

    for fp in normalized_files:
        dfm = pd.read_csv(fp)
        dfm_clean = drop_missing_tests(dfm, to_drop)
        outf = fp.replace('_normalized.csv','_final.csv')
        dfm_clean.to_csv(outf, index=False)
        print(f"Wrote final: {outf} (dropped {len(dfm)-len(dfm_clean)} rows)")
