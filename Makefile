.PHONY: data

data: data/tiffs/bground_bearbox.tiff data/tiffs/bground_bucket.tiff data/tiffs/bground_chamber_gradient.tiff \
data/tiffs/bground_cross.tiff data/tiffs/bground_odorbox.tiff data/tiffs/roi_bearbox_01.tiff data/tiffs/roi_bucket_01.tiff \
data/tiffs/roi_chamber_01.tiff data/tiffs/roi_cross_01.tiff data/tiffs/roi_odorbox_01.tiff data/proc/results_00.h5 \
data/proc/results_00.yaml data/proc/roi_00.tiff data/metadata.json data/config.yaml data/depth_ts.txt data/test-out.avi

data/metadata.json:
	aws s3 cp s3://moseq2-testdata/extract/metadata.json data/ --request-payer=requester

data/config.yaml:
	aws s3 cp s3://moseq2-testdata/pca/config.yaml data/ --request-payer=requester

data/depth_ts.txt:
	aws s3 cp s3://moseq2-testdata/extract/depth_ts.txt data/ --request-payer=requester

data/test-out.avi:
	aws s3 cp s3://moseq2-testdata/extract/test-out.avi data/ --request-payer=requester

data/tiffs/bground_bearbox.tiff:
	aws s3 cp s3://moseq2-testdata/extract/tiffs/ data/tiffs/ --request-payer=requester --recursive

data/proc/results_00.h5:
	aws s3 cp s3://moseq2-testdata/pca/proc/ data/proc/ --request-payer=requester --recursive
