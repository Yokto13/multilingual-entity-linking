from collections import defaultdict
import json
from operator import add
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

conf = pyspark.SparkConf()
# conf.set("spark.hadoop.io.compression.codecs", "io.sensesecure.hadoop.xz.XZCodec")
sc = pyspark.SparkContext(conf=conf)

spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()

damuel_workdir = Path("/home/farhand/damuel_no_spanish")
# damuel_workdir = Path("/home/farhand/damuel_test_spark")
# damuel_workdir = Path("/home/farhand/damuel_spark_workdir")

language_count = defaultdict(int)
language_qid_counts = defaultdict(dict)


def jsonify(l):
    print("jsonify:", l)
    return json.loads(l)


def get_links_from_wiki(wiki):
    for l in wiki["links"]:
        if "qid" not in l:
            continue
        if l["origin"] != "wiki":
            continue
        yield int(l["qid"][1:])


def get_mentions(path):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    # filter wiki
    data = data.filter(lambda x: "wiki" in x)
    wikis = data.map(lambda x: x["wiki"])
    return wikis.flatMap(get_links_from_wiki)


def is_damuel_specific_dir(path):
    return "damuel" in path.name and "wikidata" not in str(path)


def get_lang_from_fp(fp):
    return fp.name.split("_")[-1]

def get_lang_qids(path):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    data = data.filter(lambda x: "wiki" in x or "description" in x) # either wikipedia or wikidata
    qids = data.map(lambda x: int(x["qid"][1:]))
    return qids


def get_all_qids():
    qids = set()
    for fp in damuel_workdir.iterdir():
        if not is_damuel_specific_dir(fp):
            continue
        qids.update(get_lang_qids(fp).collect())
    return qids


if __name__ == "__main__":
    qids = get_all_qids()
    for fp in damuel_workdir.iterdir():
        if not is_damuel_specific_dir(fp):
            continue
        print(fp)
        damuel_mentions = get_mentions(fp)
        lang_name = get_lang_from_fp(fp)
        language_count[lang_name] = damuel_mentions.count()
        for_save = damuel_mentions.map(lambda x: (x, 1)).reduceByKey(add).toDF(["qid", "count"])
        for_save.write.json(f"outputs/damuel_mentions_{lang_name}.json")
    
    import json
    with open("outputs/language_counts.json", "w") as f:
        json.dump(language_count, f)
