.PHONY: data

data: data/bground_bearbox.tiff data/bground_bucket.tiff data/bground_chamber_gradient.tiff \
data/bground_cross.tiff data/bground_odorbox.tiff data/roi_bearbox_01.tiff data/roi_bucket_01.tiff \
data/roi_chamber_01.tiff data/roi_cross_01.tiff data/roi_odorbox_01.tiff

data/bground_bearbox.tiff:
	aws s3 cp s3://moseq2-testdata/extract/ data/ --request-payer=requester --recursive