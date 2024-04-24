# manually test that workflows run through to completion with different input options

WORKFLOWS="
diploid 
diploid-sans-meta 
diploid-sans-hapmap
diploid-sans-mask
haploid 
haploid-sans-meta 
haploid-sans-hapmap
haploid-sans-mask
"

for WORKFLOW in $WORKFLOWS; do
  snakemake -c10 --configfile=.test-workflow/$WORKFLOW/test.yaml &>test-workflow-$WORKFLOW.log
done
