/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright 2017-2019 University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/
#pragma once

#include "../common/AudioClient.hpp"
#include "../common/BufferedProcess.hpp"
#include "../common/FluidBaseClient.hpp"
#include "../common/FluidNRTClientWrapper.hpp"
#include "../common/ParameterConstraints.hpp"
#include "../common/ParameterSet.hpp"
#include "../common/ParameterTrackChanges.hpp"
#include "../common/ParameterTypes.hpp"
#include "../../algorithms/public/Envelope.hpp"
#include <tuple>

namespace fluid {
namespace client {
namespace ampfeature {

enum AmpFeatureParamIndex {
  kSelect,
  kFastRampUpTime,
  kFastRampDownTime,
  kSlowRampUpTime,
  kSlowRampDownTime,
  kSilenceThreshold,
  kHiPassFreq,
};

constexpr auto AmpFeatureParams = defineParameters(
    ChoicesParam("select","Selection of Outputs","diff","fast","slow"),
    LongParam("fastRampUp", "Fast Envelope Ramp Up Length", 1, Min(1)),
    LongParam("fastRampDown", "Fast Envelope Ramp Down Length", 1, Min(1)),
    LongParam("slowRampUp", "Slow Envelope Ramp Up Length", 100, Min(1)),
    LongParam("slowRampDown", "Slow Envelope Ramp Down Length", 100, Min(1)),
    FloatParam("floor", "Floor value (dB)", -145, Min(-144), Max(144)),
    FloatParam("highPassFreq", "High-Pass Filter Cutoff", 85, Min(0)));

class AmpFeatureClient : public FluidBaseClient, public AudioIn, public AudioOut
{

  static constexpr index mMaxFeatures = 3;

public:
  using ParamDescType = decltype(AmpFeatureParams);

  using ParamSetViewType = ParameterSetView<ParamDescType>;
  std::reference_wrapper<ParamSetViewType> mParams;

  void setParams(ParamSetViewType& p) { mParams = p; }

  template <size_t N>
  auto& get() const
  {
    return mParams.get().template get<N>();
  }

  static constexpr auto& getParameterDescriptors() { return AmpFeatureParams; }

  AmpFeatureClient(ParamSetViewType& p) : mParams(p)
  {
    audioChannelsIn(1);
    audioChannelsOut(mMaxFeatures);
    FluidBaseClient::setInputLabels({"audio input"});
    FluidBaseClient::setOutputLabels({"1 when slice detected, 0 otherwise"});
  }

  template <typename T>
  void process(std::vector<HostVector<T>>& input,
               std::vector<HostVector<T>>& output, FluidContext&)
  {

    if (!input[0].data() || !output[0].data()) return;

    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);

    if (!mAlgorithm.initialized())
    {
      mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq);
    }
    
    auto selection = get<kSelect>();
    index numSelected = asSigned(selection.count());
    index numOuts = std::min<index>(mMaxFeatures,numSelected);
    audioChannelsOut(numOuts);

    for (index i = 0; i < input[0].size(); i++)
    {
      double fast, slow;
      std::tie(fast, slow) = mAlgorithm.processSampleSeparate(
          input[0](i), get<kSilenceThreshold>(), get<kFastRampUpTime>(),
          get<kSlowRampUpTime>(), get<kFastRampDownTime>(),
          get<kSlowRampDownTime>(), hiPassFreq);
      index j = 0;
      if (selection[0])
        output[j++](i) = static_cast<T>(fast-slow);
      if (selection[1])
        output[j++](i) = static_cast<T>(fast);
      if (selection[2])
        output[j++](i) = static_cast<T>(slow);
    }
  }
  index latency() { return 0; }

  void reset()
  {
    double hiPassFreq = std::min(get<kHiPassFreq>() / sampleRate(), 0.5);
    mAlgorithm.init(get<kSilenceThreshold>(), hiPassFreq);
  }

private:
  algorithm::Envelope mAlgorithm;
};

template <typename HostMatrix, typename HostVectorView>
struct NRTAmpFeature
{
  template <typename Client, typename InputList, typename OutputList>
  static Result process(Client& client, InputList& inputBuffers,
                        OutputList& outputBuffers, index nFrames, index nChans,
                        std::pair<index, index> userPadding, FluidContext& c)
  {
    // expecting a single buffer in and a single buffer out
    assert(inputBuffers.size() == 1);
    assert(outputBuffers.size() == 1);

    index maxOuts = client.audioChannelsOut();

    HostMatrix inputData(nChans, nFrames);
    HostMatrix outputData(maxOuts * nChans, nFrames);

    // ignoring userPadding as AmpFeature has no padding options and no latency

    double sampleRate{0};

    for (index i = 0; i < nChans; ++i)
    {
      BufferAdaptor::ReadAccess thisInput(inputBuffers[0].buffer);
      sampleRate = thisInput.sampleRate();
      inputData.row(i)(Slice(0, nFrames)) <<=
            thisInput.samps(inputBuffers[0].startFrame, nFrames,
                            inputBuffers[0].startChan + i);
      std::vector<HostVectorView> input{inputData.row(i)};

      std::vector<HostVectorView> outputs;
      outputs.reserve(maxOuts);
      for(index j = 0; j < maxOuts; ++j)
        outputs.emplace_back(outputData.row(i * maxOuts + j));

      client.reset();
      client.process(input, outputs, c);
    }


    BufferAdaptor::Access thisOutput(outputBuffers[0]);
    index numOuts = client.audioChannelsOut();
    Result                r = thisOutput.resize(nFrames, nChans * numOuts, sampleRate);
    if (!r.ok()) return r;
    for (index j = 0; j < nChans; ++j) {
      for (index k = 0; k < numOuts; ++k)
      thisOutput.samps(j * numOuts + k) <<=
        outputData.row(j * maxOuts + k)(Slice(0, nFrames));
    }

    return {};
  }
};
} // namespace ampfeature

using RTAmpFeatureClient = ClientWrapper<ampfeature::AmpFeatureClient>;

auto constexpr NRTAmpFeatureParams = makeNRTParams<ampfeature::AmpFeatureClient>(
    InputBufferParam("source", "Source Buffer"),
    BufferParam("features", "Feature Buffer"));

using NRTAmpFeatureClient =
    impl::NRTClientWrapper<ampfeature::NRTAmpFeature, ampfeature::AmpFeatureClient, decltype(NRTAmpFeatureParams),
                    NRTAmpFeatureParams, 1, 1>;

using NRTThreadedAmpFeatureClient = NRTThreadingAdaptor<NRTAmpFeatureClient>;

} // namespace client
} // namespace fluid
