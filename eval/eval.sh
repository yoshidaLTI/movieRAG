echo "[]" > movie/大分大学入門/qg_result.json

# 知識
python qg/qg_direct.py  movie/大分大学入門/result.json --by-time --bloom-level 知識
python qg/qg_rag.py     movie/大分大学入門/result.json --by-time --bloom-level 知識 --rag-k 2
python qg/qg_rag_cot.py movie/大分大学入門/result.json --by-time --bloom-level 知識 --rag-k 2

# 応用
python qg/qg_direct.py  movie/大分大学入門/result.json --by-time --bloom-level 応用
python qg/qg_rag.py     movie/大分大学入門/result.json --by-time --bloom-level 応用 --rag-k 3
python qg/qg_rag_cot.py movie/大分大学入門/result.json --by-time --bloom-level 応用 --rag-k 3

# 評価
python qg/qg_direct.py  movie/大分大学入門/result.json --by-time --bloom-level 評価
python qg/qg_rag.py     movie/大分大学入門/result.json --by-time --bloom-level 評価 --rag-k 4
python qg/qg_rag_cot.py movie/大分大学入門/result.json --by-time --bloom-level 評価 --rag-k 4

python eval/eval_qg_quality.py movie/大分大学入門/qg_result.json


python extract/rag_build.py movie/情報ネットワーク/result.json

echo "[]" > movie/情報ネットワーク/qg_result.json

# 知識
python qg/qg_direct.py  movie/情報ネットワーク/result.json --by-time --bloom-level 知識
python qg/qg_rag.py     movie/情報ネットワーク/result.json --by-time --bloom-level 知識 --rag-k 2
python qg/qg_rag_cot.py movie/情報ネットワーク/result.json --by-time --bloom-level 知識 --rag-k 2

# 応用
python qg/qg_direct.py  movie/情報ネットワーク/result.json --by-time --bloom-level 応用
python qg/qg_rag.py     movie/情報ネットワーク/result.json --by-time --bloom-level 応用 --rag-k 3
python qg/qg_rag_cot.py movie/情報ネットワーク/result.json --by-time --bloom-level 応用 --rag-k 3

# 評価
python qg/qg_direct.py  movie/情報ネットワーク/result.json --by-time --bloom-level 評価
python qg/qg_rag.py     movie/情報ネットワーク/result.json --by-time --bloom-level 評価 --rag-k 4
python qg/qg_rag_cot.py movie/情報ネットワーク/result.json --by-time --bloom-level 評価 --rag-k 4

python eval/eval_qg_quality.py movie/情報ネットワーク/qg_result.json