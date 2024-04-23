# manually test workflows with different inputs

snakemake -c10 --configfile .test-workflow/diploid/test.yaml &>test-workflow-diploid.log
snakemake -c10 --configfile .test-workflow/diploid-sans-meta/test.yaml &>test-workflow-diploid-sans-meta.log
snakemake -c10 --configfile .test-workflow/diploid-sans-hapmap/test.yaml &>test-workflow-diploid-sans-hapmap.log
snakemake -c10 --configfile .test-workflow/diploid-sans-mask/test.yaml &>test-workflow-diploid-sans-mask.log

#snakemake -c10 --configfile .test-workflow/haploid/test.yaml &>test-workflow-haploid.log
#snakemake -c10 --configfile .test-workflow/haploid-sans-meta/test.yaml &>test-workflow-haploid-sans-meta.log
#snakemake -c10 --configfile .test-workflow/haploid-sans-hapmap/test.yaml &>test-workflow-haploid-sans-hapmap.log
#snakemake -c10 --configfile .test-workflow/haploid-sans-mask/test.yaml &>test-workflow-haploid-sans-mask.log
