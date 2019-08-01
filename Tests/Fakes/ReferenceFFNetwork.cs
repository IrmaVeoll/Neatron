using System;

namespace Tests.Fakes
{
    public class ReferenceFFNetwork
    {
        private readonly int _numInput; // number input nodes
        private readonly int _numHidden;
        private readonly int _numOutput;

        private readonly float[] _inputs;
        private readonly float[][] _ihWeights; // input-hidden
        private readonly float[] _hBiases;
        private readonly float[] _hOutputs;

        private readonly float[][] _hoWeights; // hidden-output
        private readonly float[] _oBiases;
        private readonly float[] _outputs;

        private readonly Random _rnd;
        private float[] _hSums;
        private float[] _oSums;

        public ReferenceFFNetwork(int numInput, int numHidden, int numOutput)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            _inputs = new float[numInput];

            _ihWeights = MakeMatrix(numInput, numHidden, 0.0f);
            _hBiases = new float[numHidden];
            _hOutputs = new float[numHidden];

            _hoWeights = MakeMatrix(numHidden, numOutput, 0.0f);
            _oBiases = new float[numOutput];
            _outputs = new float[numOutput];

            _rnd = new Random(0);
            InitializeWeights(); // all weights and biases
        } // ctor

        private static float[][] MakeMatrix(int rows,
            int cols, float v) // helper for ctor, Train
        {
            var result = new float[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new float[cols];
            for (var i = 0; i < rows; ++i)
            for (var j = 0; j < cols; ++j)
                result[i][j] = v;
            return result;
        }

        private void InitializeWeights() // helper for ctor
        {
            // initialize weights and biases to small random values
            var numWeights = (_numInput * _numHidden) +
                             (_numHidden * _numOutput) + _numHidden + _numOutput;
            var initialWeights = new float[numWeights];
            for (var i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001f - 0.0001f) * (float) _rnd.NextDouble() + 0.0001f;
            SetWeights(initialWeights);
        }

        public void SetWeights(float[] weights)
        {
            // copy serialized weights and biases in weights[] array
            // to i-h weights, i-h biases, h-o weights, h-o biases
            var numWeights = (_numInput * _numHidden) +
                             (_numHidden * _numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            var k = 0; // points into weights param

            for (var i = 0; i < _numInput; ++i)
            for (var j = 0; j < _numHidden; ++j)
                _ihWeights[i][j] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
                _hBiases[i] = weights[k++];
            for (var i = 0; i < _numHidden; ++i)
            for (var j = 0; j < _numOutput; ++j)
                _hoWeights[i][j] = weights[k++];
            for (var i = 0; i < _numOutput; ++i)
                _oBiases[i] = weights[k++];
        }

        public float[] GetWeights()
        {
            var numWeights = (_numInput * _numHidden) +
                             (_numHidden * _numOutput) + _numHidden + _numOutput;
            var result = new float[numWeights];
            var k = 0;
            for (var i = 0; i < _ihWeights.Length; ++i)
            for (var j = 0; j < _ihWeights[0].Length; ++j)
                result[k++] = _ihWeights[i][j];
            for (var i = 0; i < _hBiases.Length; ++i)
                result[k++] = _hBiases[i];
            for (var i = 0; i < _hoWeights.Length; ++i)
            for (var j = 0; j < _hoWeights[0].Length; ++j)
                result[k++] = _hoWeights[i][j];
            for (var i = 0; i < _oBiases.Length; ++i)
                result[k++] = _oBiases[i];
            return result;
        }

        public float[] ComputeOutputs(float[] xValues)
        {
            _hSums = new float[_numHidden]; // hidden nodes sums scratch array
            _oSums = new float[_numOutput]; // output nodes sums

            for (var i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                _inputs[i] = xValues[i];
            // note: no need to copy x-values unless you implement a ToString.
            // more efficient is to simply use the xValues[] directly.

            for (var j = 0; j < _numHidden; ++j) // compute i-h sum of weights * inputs
            for (var i = 0; i < _numInput; ++i)
                _hSums[j] += _inputs[i] * _ihWeights[i][j]; // note +=

            for (var i = 0; i < _numHidden; ++i) // add biases to hidden sums
                _hSums[i] += _hBiases[i];

            for (var i = 0; i < _numHidden; ++i) // apply activation
                _hOutputs[i] = SoftSign(_hSums[i]); // hard-coded

            for (var j = 0; j < _numOutput; ++j) // compute h-o sum of weights * hOutputs
            for (var i = 0; i < _numHidden; ++i)
                _oSums[j] += _hOutputs[i] * _hoWeights[i][j];

            for (var i = 0; i < _numOutput; ++i) // add biases to output sums
                _oSums[i] += _oBiases[i];

            for (var i = 0; i < _numOutput; ++i) // apply log-sigmoid activation
                _outputs[i] = SoftSign(_oSums[i]);

            var retResult = new float[_numOutput]; // could define a GetOutputs 
            Array.Copy(_outputs, retResult, retResult.Length);
            return retResult;
        }

        private static float SoftSign(float x) => x / (0.5f + MathF.Abs(x));
        private static float SoftSignDerivative(float x) => 0.5f / MathF.Pow((0.5f + MathF.Abs(x)), 2);

        public float[] Train(float[][] trainData, float learnRate, int maxEpochs = 1, float momentum = 0.0f)
        {
            // train using back-prop
            // back-prop specific arrays
            var hoGrads = MakeMatrix(_numHidden, _numOutput, 0.0f); // hidden-to-output weight gradients
            var obGrads = new float[_numOutput]; // output bias gradients

            var ihGrads = MakeMatrix(_numInput, _numHidden, 0.0f); // input-to-hidden weight gradients
            var hbGrads = new float[_numHidden]; // hidden bias gradients

            var oSignals =
                new float[_numOutput]; // local gradient output signals - gradients w/o associated input terms
            var hSignals = new float[_numHidden]; // local gradient hidden node signals

            // back-prop momentum specific arrays 
            var ihPrevWeightsDelta = MakeMatrix(_numInput, _numHidden, 0.0f);
            var hPrevBiasesDelta = new float[_numHidden];
            var hoPrevWeightsDelta = MakeMatrix(_numHidden, _numOutput, 0.0f);
            var oPrevBiasesDelta = new float[_numOutput];

            var epoch = 0;
            var xValues = new float[_numInput]; // inputs
            var tValues = new float[_numOutput]; // target values

            var sequence = new int[trainData.Length];
            for (var i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            var errInterval = maxEpochs / 10; // interval to check error
            errInterval = errInterval == 0 ? 1 : errInterval;
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    var trainErr = Error(trainData);
                    Console.WriteLine("epoch = " + epoch + "  error = " +
                                      trainErr.ToString("F4"));
                    //Console.ReadLine();
                }

                Shuffle(sequence); // visit each training data in random order
                for (var ii = 0; ii < trainData.Length; ++ii)
                {
                    var idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, _numInput);
                    Array.Copy(trainData[idx], _numInput, tValues, 0, _numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs 

                    // indices: i = inputs, j = hiddens, k = outputs

                    // 1. compute output node signals
                    float derivative;
                    for (var k = 0; k < _numOutput; ++k)
                    {
                        var errorSignal = tValues[k] - _outputs[k];
                        derivative = SoftSignDerivative(_oSums[k]);
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. compute hidden-to-output weight gradients using output signals
                    for (var j = 0; j < _numHidden; ++j)
                    for (var k = 0; k < _numOutput; ++k)
                        hoGrads[j][k] = oSignals[k] * _hOutputs[j];

                    // 2b. compute output bias gradients using output signals
                    for (var k = 0; k < _numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0f; // dummy assoc. input value

                    // 3. compute hidden node signals
                    for (var j = 0; j < _numHidden; ++j)
                    {
                        derivative = SoftSignDerivative(_hSums[j]);
                        var sum = 0.0f; // need sums of output signals times hidden-to-output weights
                        for (var k = 0; k < _numOutput; ++k)
                        {
                            sum += oSignals[k] * _hoWeights[j][k]; // represents error signal
                        }

                        hSignals[j] = derivative * sum;
                    }

                    // 4. compute input-hidden weight gradients
                    for (var i = 0; i < _numInput; ++i)
                    for (var j = 0; j < _numHidden; ++j)
                        ihGrads[i][j] = hSignals[j] * _inputs[i];

                    // 4b. compute hidden node bias gradients
                    for (var j = 0; j < _numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0f; // dummy 1.0 input

                    // == update weights and biases

                    // update input-to-hidden weights
                    for (var i = 0; i < _numInput; ++i)
                    {
                        for (var j = 0; j < _numHidden; ++j)
                        {
                            var delta = ihGrads[i][j] * learnRate;
                            _ihWeights[i][j] += delta; // would be -= if (o-t)
                            _ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // save for next time
                        }
                    }

                    // update hidden biases
                    for (var j = 0; j < _numHidden; ++j)
                    {
                        var delta = hbGrads[j] * learnRate;
                        _hBiases[j] += delta;
                        _hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // update hidden-to-output weights
                    for (var j = 0; j < _numHidden; ++j)
                    {
                        for (var k = 0; k < _numOutput; ++k)
                        {
                            var delta = hoGrads[j][k] * learnRate;
                            _hoWeights[j][k] += delta;
                            _hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // update output node biases
                    for (var k = 0; k < _numOutput; ++k)
                    {
                        var delta = obGrads[k] * learnRate;
                        _oBiases[k] += delta;
                        _oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // while

            var bestWts = GetWeights();
            return bestWts;
        } // Train

        private void Shuffle(int[] sequence) // instance method
        {
            for (var i = 0; i < sequence.Length; ++i)
            {
                var r = _rnd.Next(i, sequence.Length);
                var tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle

        private float Error(float[][] trainData)
        {
            // average squared error per training item
            var sumSquaredError = 0.0f;
            var xValues = new float[_numInput]; // first numInput values in trainData
            var tValues = new float[_numOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (var i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, _numInput);
                Array.Copy(trainData[i], _numInput, tValues, 0, _numOutput); // get target values
                var yValues = ComputeOutputs(xValues); // outputs using current weights
                for (var j = 0; j < _numOutput; ++j)
                {
                    var err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainData.Length;
        } // MeanSquaredError

        public float Accuracy(float[][] testData)
        {
            // percentage correct using winner-takes all
            var numCorrect = 0;
            var numWrong = 0;
            var xValues = new float[_numInput]; // inputs
            var tValues = new float[_numOutput]; // targets

            for (var i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, _numInput); // get x-values
                Array.Copy(testData[i], _numInput, tValues, 0, _numOutput); // get t-values
                var yValues = ComputeOutputs(xValues); // computed Y
                var maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
                var tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return numCorrect * 1.0f / (numCorrect + numWrong);
        }

        private static int MaxIndex(float[] vector) // helper for Accuracy()
        {
            // index of largest value
            var bigIndex = 0;
            var biggestVal = vector[0];
            for (var i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }

            return bigIndex;
        }
    }
}