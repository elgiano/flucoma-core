#pragma once

#include <data/FluidTensor.hpp>
#include <data/TensorTypes.hpp>

#include <algorithms/public/STFT.hpp>
#include <clients/common/FluidSink.hpp>
#include <clients/common/FluidSource.hpp>

namespace fluid {
namespace client {

template<typename T>
using HostVector = FluidTensorView<T,1>;

template<typename T>
using HostMatrix = FluidTensorView<T,2>;

class BufferedProcess {
public:
  template <typename F>
  void process(std::size_t windowSize, std::size_t hopSize, F processFunc) {

    assert(windowSize <= maxWindowSize() && "Window bigger than maximum");
    for (; mFrameTime < mHostSize; mFrameTime += hopSize) {
      if (mSource) {
        RealMatrixView windowIn = mFrameIn(Slice(0), Slice(0, windowSize));
        RealMatrixView windowOut =
            mFrameOut(Slice(0), Slice(0, windowSize));
        mSource->pull(windowIn, mFrameTime);
        processFunc(windowIn, windowOut);
        if (mSink)
          mSink->push(windowOut, mFrameTime);
      }
      
    }
    mFrameTime = mFrameTime < mHostSize ? mFrameTime : mFrameTime - mHostSize;
  }

  std::size_t hostSize() const noexcept { return mHostSize; }
  void hostSize(std::size_t size) noexcept { mHostSize = size; }
  
  std::size_t maxWindowSize() const noexcept { return mFrameIn.cols(); }
  void maxSize(std::size_t frames, std::size_t channelsIn,
               std::size_t channelsOut) {
    if (channelsIn > mFrameIn.rows() || frames > mFrameIn.cols())
      mFrameIn.resize(channelsIn, frames);
    if (channelsOut > mFrameOut.rows() || frames > mFrameOut.cols())
      mFrameOut.resize(channelsOut, frames);
  }

  void setBuffers(FluidSource<double> &source, FluidSink<double> &sink) {
    mSource = &source;
    mSink = &sink;
  }

private:
  std::size_t mFrameTime = 0 ;
  std::size_t mHostSize;
  RealMatrix mFrameIn;
  RealMatrix mFrameOut;
  FluidSource<double> *mSource = nullptr;
  FluidSink<double> *mSink = nullptr;
};

template <typename T, typename U, typename Client, size_t maxWinParam,
          size_t winParam, size_t hopParam, size_t FFTParam, bool normalise>
class STFTBufferedProcess {
  using HostVector = HostVector<U>;

public:
  template <typename F>
  void process(Client &x, std::vector<HostVector> &input,
               std::vector<HostVector> &output, F &&processFunc) {

    size_t winSize = x.template get<winParam>();
    size_t hopSize = x.template changed<hopParam>() ? x.template get<hopParam>()
                                                    : winSize / 2;
    size_t fftSize =
        x.template get<FFTParam>() != -1 ? x.template get<FFTParam>() : winSize;

    // TODO: constraints check here: error and bail if unmet

    bool newParams = paramsChanged(winSize, hopSize, fftSize);

    if (!mSTFT.get() || newParams)
      mSTFT.reset(new algorithm::STFT(winSize, fftSize, hopSize));

    if (!mISTFT.get() || newParams)
      mISTFT.reset(new algorithm::ISTFT(winSize, fftSize, hopSize));

    if (!input[0].data())
      return; // if there's not actually an audio input, no point continuing

    std::size_t chansIn = x.audioChannelsIn();
    std::size_t chansOut = x.audioChannelsOut();

    assert(chansIn == input.size());
    assert(chansOut == output.size());

    size_t hostBufferSize = input[0].size();
    mBufferedProcess.hostSize(hostBufferSize); // safe assumption?
    mInputBuffer.setHostBufferSize(hostBufferSize);
    mOutputBuffer.setHostBufferSize(hostBufferSize);

    std::size_t maxWin = x.template get<maxWinParam>();
    mInputBuffer.setSize(maxWin);
    mOutputBuffer.setSize(maxWin);
    mInputBuffer.reset(chansIn);
    // TODO: make explicit the extra channel for post-normalisation
    mOutputBuffer.reset(chansOut + 1);
    mBufferedProcess.maxSize(maxWin, chansIn, chansOut + 1);

    if (std::max(maxWin,hostBufferSize) > mFrameAndWindow.cols())
      mFrameAndWindow.resize(chansOut + 1, std::max(maxWin,hostBufferSize));
    
    if ((fftSize / 2 + 1) != mSpectrumIn.cols())
    {
      mSpectrumIn.resize(chansIn, (fftSize / 2 + 1));
    }
    
    if ((fftSize / 2 + 1) != mSpectrumOut.cols())
    {
      mSpectrumOut.resize(chansOut, (fftSize / 2 + 1));
    }


    mBufferedProcess.setBuffers(mInputBuffer, mOutputBuffer);

    mInputBuffer.push(HostMatrix<U>(input[0]));

    mBufferedProcess.process(
        winSize, hopSize,
        [this, &processFunc, chansIn, chansOut](RealMatrixView in, RealMatrixView out) {
          for(int i = 0; i < chansIn; ++i)
            mSTFT->processFrame(in.row(i), mSpectrumIn.row(i));
          processFunc(mSpectrumIn, mSpectrumOut);
          for(int i = 0; i < chansOut; ++i)
            mISTFT->processFrame(mSpectrumOut.row(i), out.row(i));
          out.row(chansOut) = mSTFT->window();
          out.row(chansOut).apply(mISTFT->window(),[](double &x, double &y) { x *= y; });
        });

    // TODO: if normalise
    if(normalise)
    {
      RealMatrixView unnormalisedFrame = mFrameAndWindow(Slice(0), Slice(0, hostBufferSize));
      mOutputBuffer.pull(unnormalisedFrame);
      for(int i = 0; i < chansOut; ++i)
      {

        unnormalisedFrame.row(i).apply(unnormalisedFrame.row(chansOut),[](double &x, double g) {
                                         if (x) {  x /= g ? g : 1; }
                                       });
        if (output[i].data())  output[i] = unnormalisedFrame.row(i);
      }
    }
  }

private:
  bool paramsChanged(std::size_t winSize, std::size_t hopSize,
                     std::size_t fftSize) {
    static std::size_t win, hop, fft;
    bool res = (win != winSize) || (hop != hopSize) || (fft != fftSize);

    win = winSize;
    hop = hopSize;
    fft = fftSize;

    return res;
  }

  RealMatrix mFrameAndWindow;
  ComplexMatrix mSpectrumIn;
  ComplexMatrix mSpectrumOut;
  std::unique_ptr<algorithm::STFT> mSTFT;
  std::unique_ptr<algorithm::ISTFT> mISTFT;
  BufferedProcess mBufferedProcess;
  FluidSource<double> mInputBuffer;
  FluidSink<double> mOutputBuffer;
};

} // namespace client
} // namespace fluid
