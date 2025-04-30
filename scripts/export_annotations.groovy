// Get the name of the current image
def imageName = getCurrentImageData().getServer().getMetadata().getName()

// Define the output path using the image name
def path = "E:\\ColorectalCancer\\analysis\\annotations_Tanja\\" + imageName + ".geojson"

// Get the annotation objects
def annotations = getAnnotationObjects()

// Export the annotations to GeoJSON
exportObjectsToGeoJson(annotations, path, "PRETTY_JSON", "FEATURE_COLLECTION")

print "Saved to " + path