OUTPUT=SMGF.log
{
    echo "ACM"
    python SMGF.py --dataset acm --knn_k 10
    echo "DBLP"
    python SMGF.py --dataset dblp --knn_k 10
    echo "IMDB"
    python SMGF.py --dataset imdb --knn_k 500
    echo "Yelp"
    python SMGF.py --dataset yelp --knn_k 200
    echo "Freebase"
    python SMGF.py --dataset freebase --knn_k 10
    echo "RM"
    python SMGF.py --dataset rm --knn_k 10
    echo "Query"
    python SMGF.py --dataset query --knn_k 20
    echo "Amazon-photos"
    python SMGF.py --dataset amazon-photos --knn_k 10
    echo "Amazon-computers"
    python SMGF.py --dataset amazon-computers --knn_k 10
    echo "MAG-eng"
    python SMGF.py --dataset mageng --knn_k 10
    echo "MAG-phy"
    python SMGF.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT

OUTPUT=SMGFQ.log
{
    echo "ACM"
    python SMGFQ.py --dataset acm --knn_k 10
    echo "DBLP"
    python SMGFQ.py --dataset dblp --knn_k 10
    echo "IMDB"
    python SMGFQ.py --dataset imdb --knn_k 500
    echo "Yelp"
    python SMGFQ.py --dataset yelp --knn_k 200
    echo "Freebase"
    python SMGFQ.py --dataset freebase --knn_k 10
    echo "RM"
    python SMGFQ.py --dataset rm --knn_k 10
    echo "Query"
    python SMGFQ.py --dataset query --knn_k 20
    echo "Amazon-photos"
    python SMGFQ.py --dataset amazon-photos --knn_k 10
    echo "Amazon-computers"
    python SMGFQ.py --dataset amazon-computers --knn_k 10
    echo "MAG-eng"
    python SMGFQ.py --dataset mageng --knn_k 10
    echo "MAG-phy"
    python SMGFQ.py --dataset magphy --knn_k 10
}|tee -a $OUTPUT
