OUTPUT=SMGF.log
{
    python SMGF_LA.py --dataset acm --knn_k 10
    python SMGF_LA.py --dataset dblp --knn_k 10
    python SMGF_LA.py --dataset imdb --knn_k 500
    python SMGF_LA.py --dataset yelp --knn_k 200
    python SMGF_LA.py --dataset freebase --knn_k 10
    python SMGF_LA.py --dataset rm --knn_k 10
    python SMGF_LA.py --dataset amazon-photos --knn_k 10
    python SMGF_LA.py --dataset amazon-computers --knn_k 10
    python SMGF_LA.py --dataset mageng --knn_k 10
    python SMGF_LA.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT

OUTPUT1=SMGF_PI.log
{
    python SMGF_PI.py --dataset acm --knn_k 10
    python SMGF_PI.py --dataset dblp --knn_k 10
    python SMGF_PI.py --dataset imdb --knn_k 500
    python SMGF_PI.py --dataset yelp --knn_k 200
    python SMGF_PI.py --dataset freebase --knn_k 10
    python SMGF_PI.py --dataset rm --knn_k 10
    python SMGF_PI.py --dataset amazon-photos --knn_k 10
    python SMGF_PI.py --dataset amazon-computers --knn_k 10
    python SMGF_PI.py --dataset mageng --knn_k 10
    python SMGF_PI.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT1
