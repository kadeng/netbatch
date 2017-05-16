
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <chrono>
#include <dlib/dir_nav.h>
#include "proto/message.pb.h"
#include "argparse.hpp" // See https://github.com/hbristow/argparse
#include <memory>
#include <dlib/dir_nav.h>

using namespace std;
using namespace netbatch;

struct FileFilter {
    string &extension;
    bool accept_hidden;

    FileFilter(std::string &extension, bool accept_hidden) : accept_hidden(accept_hidden), extension(extension){

    }

    bool operator()(dlib::file &f) const {
        if (f.name().size()==0) {
            return false;
        }
        if (!accept_hidden && f.name()[0]=='.') {
            return false;
        }
        if (extension.size()>0) {
            return f.name().rfind(extension) == f.name().size()-extension.size();
        }
        return true;
    }

};


int main(int argc, const char *argv[])
{
    ArgumentParser parser;

    // add some arguments to search for
    parser.appName("recordfiler");
    parser.addArgument("-r", "--resultpath", 1, true);
    parser.addArgument("-e", "--file_extension", 1, false);
    parser.addArgument("-a", "--append", 1, true);

    parser.addFinalArgument("sourcedir");
    // parse the command-line arguments - throws if invalid format
    parser.parse(argc, argv);

    // if we get here, the configuration is valid
    string sourcedir = parser.retrieve<string>("sourcedir");
    string resultpath = sourcedir;
    string extension = parser.retrieve<string>("file_extension");
    if (parser.retrieve<string>("resultpath").size()>0) {
        resultpath = parser.retrieve<string>("resultpath");
    }
    dlib::directory srcdir(sourcedir);

    vector<dlib::file> files = dlib::get_files_in_directory_tree(srcdir, FileFilter(extension, false), 6);
    sort(files.begin(), files.end());

    for (int i=0;i<files.size();i++) {
        cout << files[i].full_name() << endl;
    }
    {
        ofstream rf;
        ofstream irf;
        if (parser.exists("append")) {
            cout << "Opening " << resultpath << ".rec and .idx for append" <<endl;
            rf.open(resultpath+".rec", std::ios::binary | std::ios::ate);
            irf.open(resultpath+".idx", std::ios::binary | std::ios::ate);
        } else {
            cout << "Opening " << resultpath << ".rec and .idx for write" <<endl;
            rf.open(resultpath, std::ios::binary | std::ios::trunc);
            irf.open(resultpath+".idx", std::ios::binary | std::ios::trunc);
        }
        if (!rf) {
            cerr << "Failed to open " << resultpath << ".rec for writing" << endl;
            return -2;
        }
        if (!irf) {
            cerr << "Failed to open " << resultpath << ".idx for writing" << endl;
            return -3;
        }
        streamsize pos = rf.tellp();

        for (int i=0;i<files.size();i++) {
            ifstream file(files[i].full_name(), std::ios::binary | std::ios::ate);
            if (file) {
                streamsize size = file.tellg();
                file.seekg(0, ios::beg);
                rf << file.rdbuf();
                streamsize newpos = rf.tellp();
                streamsize oldirfpos = irf.tellp();
                irf.write((const char*) &newpos, sizeof(streamsize));
                pos = newpos;
                cout << "Wrote " << sourcedir << " into " << resultpath << ".rec" << endl;

            } else {
                cout << "Failed to open " << files[i].full_name() << " for reading" << endl;
            }
        }
        irf.flush();
        rf.flush();
        cout << "Finished writing  " << files.size() << " Files of " << pos << " bytes total" << endl;
        cout << "Final size of record file  " << resultpath << ".rec " << rf.tellp() << " bytes." << endl;
        cout << "Final size of index file  " << resultpath << ".idx " << irf.tellp() << " bytes." << endl;

    }
    return EXIT_SUCCESS;
}
