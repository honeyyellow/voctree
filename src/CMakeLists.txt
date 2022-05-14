set(NAME engine)

set(SOURCES
        Catalog.cu
        Catalog.h
        Configuration.cu
        Configuration.h
        Database.cu
        Database.h
        ExtKmeans.cu
        ExtKmeans.h
        FeatureMethod.cu
        FeatureMethod.h
        FileHelper.cu
        FileHelper.h
        FileManager.cu
        FileManager.h
        KeyPointPersistor.cu
        KeyPointPersistor.h
        KMeans.cu
        KMeans.h
        main.cu
        Matching.cu
        Matching.h
        MatPersistor.cu
        MatPersistor.h
	PopSiftDetector.cu
	PopSiftDetector.h
        Server.cu
        Server.h
        ShootSegmenter.cu
        ShootSegmenter.h
        VecPersistor.h
        VocTree.cu
        VocTree.h
) # SOURCES

add_executable(${NAME} ${SOURCES})
set_target_properties(${NAME} PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(${NAME} PUBLIC PopSift::popsift ${OpenCV_LIBS} nvToolsExt)

# target_link_libraries(${NAME} ${OpenCV_LIBS})

