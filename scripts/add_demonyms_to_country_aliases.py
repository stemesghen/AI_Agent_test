import pandas as pd

# ---- FIXED: load correct file ----
df = pd.read_csv("data/country_aliases.csv")

# The demonym → ISO2 mapping
demonym_map = {

    # Major actors in sanctions / maritime news
    "russian": "RU",
    "chinese": "CN",
    "greek": "GR",
    "turkish": "TR",
    "spanish": "ES",
    "italian": "IT",
    "french": "FR",
    "german": "DE",
    "british": "GB",
    "japanese": "JP",
    "korean": "KR",
    "north korean": "KP",
    "south korean": "KR",
    "iranian": "IR",
    "iraqi": "IQ",
    "syrian": "SY",
    "israeli": "IL",
    "saudi": "SA",
    "saudi arabian": "SA",
    "emirati": "AE",
    "qatari": "QA",
    "kuwaiti": "KW",

    # Americas
    "american": "US",
    "u.s.": "US",
    "us": "US",
    "canadian": "CA",
    "mexican": "MX",
    "colombian": "CO",
    "venezuelan": "VE",
    "brazilian": "BR",
    "argentine": "AR",
    "argentinian": "AR",
    "chilean": "CL",
    "peruvian": "PE",

    # Europe
    "dutch": "NL",
    "polish": "PL",
    "swedish": "SE",
    "norwegian": "NO",
    "danish": "DK",
    "finnish": "FI",
    "austrian": "AT",
    "hungarian": "HU",
    "romanian": "RO",
    "bulgarian": "BG",
    "portuguese": "PT",
    "czech": "CZ",
    "slovak": "SK",
    "slovene": "SI",
    "slovenian": "SI",
    "croatian": "HR",
    "serbian": "RS",
    "bosnian": "BA",
    "georgian": "GE",
    "armenian": "AM",
    "azerbaijani": "AZ",

    # Africa
    "egyptian": "EG",
    "libyan": "LY",
    "tunisian": "TN",
    "algerian": "DZ",
    "moroccan": "MA",
    "ethiopian": "ET",
    "eritrean": "ER",
    "kenyan": "KE",
    "tanzanian": "TZ",
    "somali": "SO",
    "south african": "ZA",
    "ghanaian": "GH",
    "nigerian": "NG",
    "angolan": "AO",
    "ivorian": "CI",
    "côte d’ivoire": "CI",

    # Central & South Asia
    "pakistani": "PK",
    "afghan": "AF",
    "indian": "IN",
    "bangladeshi": "BD",
    "sri lankan": "LK",
    "nepali": "NP",
    "kazakh": "KZ",
    "uzbek": "UZ",
    "tajik": "TJ",
    "kyrgyz": "KG",

    # East & SE Asia
    "vietnamese": "VN",
    "thai": "TH",
    "laotian": "LA",
    "burmese": "MM",
    "myanmar": "MM",
    "cambodian": "KH",
    "filipino": "PH",
    "philippine": "PH",
    "malaysian": "MY",
    "indonesian": "ID",
    "taiwanese": "TW",
    "hong kong": "HK",
    "macanese": "MO",

    # Oceania
    "australian": "AU",
    "new zealander": "NZ",
    "kiwi": "NZ",

    # Middle East (additional)
    "yemeni": "YE",
    "omani": "OM",
    "bahraini": "BH",
    "lebanese": "LB",
    "jordanian": "JO",

    # Special geopolitical designations
    "palestinian": "PS",
    "kosovar": "XK",
    "kosovan": "XK",
}

# Build new rows
rows = []
for alias, iso in demonym_map.items():
    rows.append({
        "alias": alias,
        "canonical_name": alias,
        "iso2": iso,
        "source": "manual-demonym"
    })

df_new = pd.DataFrame(rows)

# Combine + remove duplicates
df_out = pd.concat([df, df_new], ignore_index=True)
df_out = df_out.drop_duplicates(subset=["alias", "iso2"])

# Save result
df_out.to_csv("data/country_aliases.csv", index=False)

print(" Demonyms added to data/country_aliases.csv successfully!")
