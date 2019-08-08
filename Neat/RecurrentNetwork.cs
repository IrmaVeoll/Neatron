using System;
using System.Collections.Generic;
using System.Linq;
using static Neat.NetworkParameters;

namespace Neat
{
    public sealed class RecurrentNetwork : INetwork
    {
        private const float BiasValue = 1;

        private readonly NetworkConnection[][] _connections;
        private readonly int _inputCount;
        private readonly float[] _currentActivationInput;
        private readonly float[] _pastActivationOutput;
        private float[][] _states;
        private float[] _currentGradient;
        private float[] _nextGradient;
        private float[][] _weightChanges;

        private readonly INeatChromosomeEncoder _encoder;
        

        internal RecurrentNetwork(NetworkConnection[][] connections, int inputCount, int outputCount,
            INeatChromosomeEncoder encoder)
        {
            _connections = connections;
            _inputCount = inputCount;

            _currentActivationInput = new float[connections.Length];
            _pastActivationOutput = new float[connections.Length];
            _currentActivationInput[0] = BiasValue;

            _encoder = encoder;

            var sensorCount = inputCount - BiasCount;

            Sensors = new ArraySegment<float>(_currentActivationInput, BiasCount, sensorCount);
            Effectors = new ArraySegment<float>(_pastActivationOutput, inputCount, outputCount);
        }

        // ReSharper disable once CollectionNeverQueried.Global
        public IList<float> Sensors { get; }

        public IReadOnlyList<float> Effectors { get; }
        

        public void Activate()
        {
            //accumulate semilinear values
            // inputs and bias
            for (var i = 0; i < _inputCount; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _currentActivationInput[connection.TargetIdx] += _pastActivationOutput[i] * connection.Weight;
                }

                _pastActivationOutput[i] = _currentActivationInput[i];
            }

            // outputs and hidden
            for (var i = _inputCount; i < _connections.Length; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _currentActivationInput[connection.TargetIdx] += _pastActivationOutput[i] * connection.Weight;
                }
            }

            //activate and swap         
            for (var i = _inputCount; i < _connections.Length; i++)
            {
                var preActivation = _currentActivationInput[i];
                var preActivationAbs = preActivation < 0 ? -preActivation : preActivation;
                _pastActivationOutput[i] = preActivationAbs > float.Epsilon
                    ? preActivation / (0.5f + preActivationAbs) //inlined zero-centered softsign
                    : 0;

                _currentActivationInput[i] = 0f;
            }
        }

        public void Train(float[][] samples, float learningRate = 0.01f, float l1Ratio = 0, float l2Ratio = 0)
        {
            if (samples.Length == 0)
                return;
            if(samples[0].Length != Sensors.Count + Effectors.Count)
                throw new ArgumentException("Incorrect dimensions of samples array");
            
            var activationsCount = samples.Length;
            var neuronsCount = _connections.Length;
            var outputsCount = Effectors.Count;
            var activatedNeuronsCount = neuronsCount - _inputCount;
            if(_states == null || _states.Length < activationsCount)
                _states = new float[activationsCount][];

            // accumulate activation inputs and results
            AccumulateData(samples, _states, activationsCount, neuronsCount, activatedNeuronsCount);

            if (_currentGradient == null)
            {
                _currentGradient = new float[activatedNeuronsCount];
            }
            else
            {
                Array.Clear(_currentGradient, 0, activatedNeuronsCount);
            }

            if (_nextGradient == null)
            {
                _nextGradient = new float[activatedNeuronsCount];
            }
            else
            {
                Array.Clear(_nextGradient, 0, activatedNeuronsCount);
            }

            var swap = _currentGradient;

            if (_weightChanges == null)
            {
                _weightChanges = new float[_connections.Length][];
                for (var i = 0; i < _connections.Length; i++)
                {
                    _weightChanges[i] = new float[_connections[i].Length];
                }
            }
            else
            {
                for (var i = 0; i < _connections.Length; i++)
                {
                    Array.Clear(_weightChanges[i], 0, _weightChanges[i].Length);
                }
            }

            for (var t = activationsCount - 1; t > 0; t--)
            {
                // calculate gradient
                CalculateGradient(t, samples, _states, _currentGradient, outputsCount, neuronsCount, activatedNeuronsCount);

                // back propagate and accumulate weight changes
                BackPropagateAndAccumulateError(t, _currentGradient, _nextGradient, _weightChanges, _states);

                //swap gradients
                _currentGradient = _nextGradient;
                _nextGradient = swap;
                swap = _currentGradient;
                Array.Clear(_nextGradient, 0, activatedNeuronsCount);
            }

            // make weight changes
            for(var i = 0; i < _connections.Length; i++)
            for (var j = 0; j < _connections[i].Length; j++)
                _connections[i][j] = GetModifiedConnection(_connections[i][j], _weightChanges[i][j], learningRate,
                    l1Ratio, l2Ratio);
            _encoder.Encode();
        }

        private void BackPropagateAndAccumulateError(int t,
            // ReSharper disable SuggestBaseTypeForParameter
            float[] currentGradient,
            float[] nextGradient, 
            float[][] weightChanges, 
            float[][] states)
        {
            for (var i = 0; i < _connections.Length; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var gradient = currentGradient[_connections[i][j].TargetIdx - _inputCount];
                    weightChanges[i][j] += gradient * states[t - 1][i];
                    if (i >= _inputCount)
                        nextGradient[i - _inputCount] += gradient * _connections[i][j].Weight;
                }
            }
        }

        private void CalculateGradient(int t,
            float[][] samples,
            float[][] states,
            float[] currentGradient,
            int outputsCount,
            int neuronsCount,
            int activatedNeuronsCount)
        {
            var i = 0;
            var sensorsCount = _inputCount - BiasCount;
            for (; i < outputsCount; i++)
            {
                currentGradient[i] +=
                    samples[t][i + sensorsCount] - states[t][i + _inputCount]; // add error on effectors
                currentGradient[i] = currentGradient[i] * states[t][neuronsCount + i]; // multiply by derivative
            }

            for (; i < activatedNeuronsCount; i++)
            {
                currentGradient[i] = currentGradient[i] * states[t][neuronsCount + i]; // multiply by derivative
            }
        }

        private void AccumulateData(float[][] samples,
            float[][] states,
            int activationsCount, int neuronsCount,
            int activatedNeuronsCount)
        {
            for (var t = 0; t < activationsCount; t++)
            {
                Array.Clear(_currentActivationInput, _inputCount, activatedNeuronsCount);
                Array.Copy(samples[t], 0, _currentActivationInput, 1, _inputCount - BiasCount);
                ActivateNoCleanup();
                states[t] = new float[neuronsCount * 2 -
                                      _inputCount]; // All past activations and pre-activation derivatives
                Array.Copy(_pastActivationOutput, states[t], neuronsCount);
                for (var j = 0; j < activatedNeuronsCount; j++)
                {
                    var x = _currentActivationInput[_inputCount + j];
                    var d = 0.5f + (x < 0 ? -x : x);
                    states[t][neuronsCount + j] = 0.5f / (d * d);
                }
            }
        }

        private void ActivateNoCleanup()
        {
            // Exactly the same as Activate() except there's no activation inputs cleanup
            // Kept as a separate method for performance reasons 

            // accumulate semilinear values
            // inputs and bias
            for (var i = 0; i < _inputCount; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _currentActivationInput[connection.TargetIdx] += _pastActivationOutput[i] * connection.Weight;
                }

                _pastActivationOutput[i] = _currentActivationInput[i];
            }

            // outputs and hidden
            for (var i = _inputCount; i < _connections.Length; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _currentActivationInput[connection.TargetIdx] += _pastActivationOutput[i] * connection.Weight;
                }
            }

            // activate and swap         
            for (var i = _inputCount; i < _connections.Length; i++)
            {
                var preActivation = _currentActivationInput[i];
                var preActivationAbs = preActivation < 0 ? -preActivation : preActivation;
                _pastActivationOutput[i] = preActivationAbs > float.Epsilon
                    ? preActivation / (0.5f + preActivationAbs) //inlined zero-centered softsign
                    : 0;
            }
        }
        
        private NetworkConnection GetModifiedConnection(NetworkConnection connection, float weightDelta,
            float learningRate, float l1Ratio, float l2Ratio)
        {
            var oldWeight = connection.Weight;
            var newWeight = oldWeight + learningRate * (weightDelta +
                                                        l1Ratio * Math.Sign(oldWeight) +
                                                        l2Ratio * oldWeight);
            return new NetworkConnection(connection.TargetIdx, newWeight);
        }
    }
}