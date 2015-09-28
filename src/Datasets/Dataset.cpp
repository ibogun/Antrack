//
//  Dataset.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/7/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Dataset.h"
#include <regex>
#include <dirent.h>


/**
 *  List only subfolders in a given folder
 *
 *  @param folder where to look for subfolders
 *
 *  @return vector of subfolders (relative paths)
 */
std::vector<std::string> Dataset::listSubFolders(std::string folder){
    using namespace std;
    const char *PATH=folder.c_str();
    regex e ("[^\\.].*");
    
    DIR *dir = opendir(PATH);
    smatch match;

    struct dirent *entry = readdir(dir);
    
    
    vector<string> r;
    while (entry != NULL)
    {
       // if (entry->d_type == DT_DIR){
            std::string fileName=std::string(entry->d_name);
            
            regex_match(fileName,match,e);
            
            if (!match.empty()) {
                //printf("%s\n", entry->d_name);
                r.push_back(fileName);
            }
       // }
        
        
        entry = readdir(dir);
    }

    std::sort(r.begin(), r.end());
    //delete PATH;
    closedir(dir);
    //delete dir;
    
    return r;
    
}

/**
 *  List images in the folder given format
 *
 *  @param imgLocation images location
 *  @param format      images format
 *
 *  @return vector with image names
 */
std::vector<std::string> Dataset::listImages(std::string imgLocation, std::string format){
    
    using namespace std;
    vector<std::string> imgList;
    std::regex e ("(.*)\\."+format);
    
    DIR *dir;
    struct dirent *ent;
    
    std::smatch match;
    
    vector<std::string> imageFilenames;
    
    if ((dir = opendir (imgLocation.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            
            
            std::string fileName=std::string(ent->d_name);
            
            std::regex_match(fileName,match,e);
            
            //std::cout << ent->d_name <<" "<< ( match.empty() ? "did not match" : "matched" ) << std::endl;
            
            if (!match.empty()) {
                imageFilenames.push_back(imgLocation+fileName);
                //std::cout<<imgLocation+fileName<<endl;
            }
            
            //printf ("%s\n", ent->d_name);
            imgList.push_back(ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("Couldnt open directory");
        //return nullptr;
    }
    std::sort(imageFilenames.begin(), imageFilenames.end());
    return imageFilenames;
}
