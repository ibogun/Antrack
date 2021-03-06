
#ifndef SRC_FEATURES_ALLFEATURES_H_
#define SRC_FEATURES_ALLFEATURES_H_

#include "Haar.h"
#include "RawFeatures.h"
#include "Histogram.h"
#include "HaarFeatureSet.h"
#include "HoG.h"
#include "HoG_PCA.h"
#include "HoGandRawFeatures.h"
#include "MultiFeature.h"

#ifdef USE_DEEP_FEATURES
#include "DeepFeatures.h"
#include "DeepPCA.h"
#endif


#endif /* SRC_FEATURES_ALLFEATURES_H_ */
