file_list = [
    # Positive examples – Rio Branco geoglyph corridor (clearly visible ditches)
    "RIB_A01_2014_laz_1.laz",
    "RIB_A01_2014_laz_2.laz",
    "RIB_A01_2014_laz_3.laz",
    "RIB_A01_2014_laz_4.laz",
    "RIB_A01_2014_laz_5.laz",
    "RIB_A01_2014_laz_6.laz",
    "RIB_A01_2014_laz_7.laz",
    "RIB_A01_2014_laz_8.laz",
    "RIB_A01_2014_laz_9.laz",
    "RIB_A01_2014_laz_10.laz",
    "RIB_A01_2014_laz_11.laz",

    # Positive examples – Humaitá fortified villages
    "HUM_A01_2013_laz_1.laz",
    "HUM_A01_2013_laz_2.laz",
    "HUM_A01_2013_laz_3.laz",
    "HUM_A01_2013_laz_4.laz",
    "HUM_A01_2013_laz_5.laz",
    "HUM_A01_2013_laz_6.laz",

    # Negative controls – same surveys, tiles that (in published work) show blank forest
    "RIB_A01_2014_laz_40.laz",
    "RIB_A01_2014_laz_41.laz",
    "RIB_A01_2014_laz_42.laz",
    "RIB_A01_2014_laz_43.laz",
    "HUM_A01_2013_laz_25.laz",
    "HUM_A01_2013_laz_26.laz",
    "HUM_A01_2013_laz_27.laz",
    "CAU_A01_2014_laz_0.laz",
    "CAU_A01_2014_laz_1.laz",
    "CAU_A01_2014_laz_2.laz",
    "CAU_A01_2014_laz_18.laz",
    "DUC_A01_2008_laz_10.laz",
    "DUC_A01_2008_laz_7.laz"
]

with open('tile_list.txt', 'w') as f:
    f.write('\n'.join(file_list))

'tile_list.txt'
