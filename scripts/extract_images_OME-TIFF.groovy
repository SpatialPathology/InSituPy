/**
 * QuPath script to export images corresponding to all the annotation bounding boxes in an image.
 * Written for QuPath v0.4.3.
 */
 
import qupath.lib.images.writers.ome.OMEPyramidWriter

def file_suffix = "__HE__histo.ome.tif"
def tilesize = 1024
def outputDownsample = 1
def pyramidscaling = 2
def compression = OMEPyramidWriter.CompressionType.ZLIB     //J2K //UNCOMPRESSED //LZW

// Export at full resolution (or change this value)
double downsample = 1.0

// Export to a subdirectory of the current project
def dir = buildPathInProject("export")
mkdirs(dir)

// Loop through annotations and export
def server = getCurrentServer()
def annotations = getAnnotationObjects()
for (def annotation in annotations) {
    def annot_name = annotation.getName()
    def roi = annotation.getROI()
    println roi.getArea()
    int height = roi.getBoundsHeight()
    int width = roi.getBoundsWidth()
    int x = roi.getBoundsX()
    int y = roi.getBoundsY()

    def request = RegionRequest.createInstance(
        server.getPath(),
        downsample,
        annotation.getROI()
    )
    def name = getCurrentImageNameWithoutExtension()
    def outputName = "${annot_name}${file_suffix}"
    def path = buildFilePath(dir, outputName)
    println path
    // writeImageRegion(server, request, path)
    new OMEPyramidWriter.Builder(server)
                                .region(x, y, width, height)
				.compression(compression)
				.parallelize()
				.tileSize(tilesize)
				.channelsInterleaved() // Usually faster
				.scaledDownsampling(outputDownsample, pyramidscaling)
				.build()
				.writePyramid(path)
}
print "Done!"