# some minimal test cases with optional inputs

mkdir -p diploid
mkdir -p diploid-sans-meta
mkdir -p diploid-sans-hapmap
mkdir -p diploid-sans-mask
mkdir -p haploid
mkdir -p haploid-sans-meta
mkdir -p haploid-sans-hapmap
mkdir -p haploid-sans-mask

python3 ../example/simulate_data.py \
  --output-prefix diploid/diploid --samples 4
echo "input-dir: '.test-data/diploid'" >diploid/test.yaml
echo "mcmc-samples: 10" >>diploid/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix diploid-sans-meta/diploid-sans-meta --samples 4 --disable-meta
echo "input-dir: '.test-data/diploid-sans-meta'" >diploid-sans-meta/test.yaml
echo "mcmc-samples: 10" >>diploid-sans-meta/test.yaml
echo "stratify-by: None" >>diploid-sans-meta/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix diploid-sans-hapmap/diploid-sans-hapmap --samples 4 --disable-hapmap
echo "input-dir: '.test-data/diploid-sans-hapmap'" >diploid-sans-hapmap/test.yaml
echo "mcmc-samples: 10" >>diploid-sans-hapmap/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix diploid-sans-mask/diploid-sans-mask --samples 4 --disable-mask
echo "input-dir: '.test-data/diploid-sans-mask'" >diploid-sans-mask/test.yaml
echo "mcmc-samples: 10" >>diploid-sans-mask/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix haploid/haploid --samples 8
echo "input-dir: '.test-data/haploid'" >haploid/test.yaml
echo "mcmc-samples: 10" >>haploid/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix haploid-sans-meta/haploid-sans-meta --samples 8 --disable-meta
echo "input-dir: '.test-data/haploid-sans-meta'" >haploid-sans-meta/test.yaml
echo "mcmc-samples: 10" >>haploid-sans-meta/test.yaml
echo "stratify-by: None" >>haploid-sans-meta/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix haploid-sans-hapmap/haploid-sans-hapmap --samples 8 --disable-hapmap
echo "input-dir: '.test-data/haploid-sans-hapmap'" >haploid-sans-hapmap/test.yaml
echo "mcmc-samples: 10" >>haploid-sans-hapmap/test.yaml

python3 ../example/simulate_data.py \
  --output-prefix haploid-sans-mask/haploid-sans-mask --samples 8 --disable-mask
echo "input-dir: '.test-data/haploid-sans-mask'" >haploid-sans-mask/test.yaml
echo "mcmc-samples: 10" >>haploid-sans-mask/test.yaml
