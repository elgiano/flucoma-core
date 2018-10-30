#include "util/audiofile.hpp"

int main(int argc, char *argv[]) {
  using fluid::audiofile::AudioFileData;
  using fluid::audiofile::readFile;
  using fluid::audiofile::writeFile;
  using std::cout;

  if (argc <= 2) {
    cout << "usage: test_audiofile in.wav out.wav\n";
    return 1;
  }

  AudioFileData data = readFile(argv[1]);
  writeFile(data, argv[2]);

  return 0;
}
