using System;
using System.Collections.Generic;
using static Neat.NetworkParameters;

namespace Neat
{
    public sealed class Network : INetwork
    {
        private const float BiasValue = 1;
        
        private readonly NetworkConnection[][] _connections;
        private readonly int _inputCount;
        private readonly float[] _preActivation;
        private readonly float[] _postActivation;
        private readonly float[] _gradient;
        
        private readonly INeatChromosomeEncoder _encoder;

        internal Network(NetworkConnection[][] connections, int inputCount, int outputCount, INeatChromosomeEncoder encoder)
        {
            _connections = connections;
            _inputCount = inputCount;
            
            _preActivation = new float[connections.Length];
            _postActivation = new float[connections.Length];
            _postActivation[0] = BiasValue;
            _gradient = new float[connections.Length]; 

            _encoder = encoder;
            
            var sensorCount = inputCount - BiasCount;
            
            Sensors = new ArraySegment<float>(_postActivation, BiasCount, sensorCount);
            Effectors = new ArraySegment<float>(_postActivation, inputCount, outputCount);
        }

        // ReSharper disable once CollectionNeverQueried.Global
        public IList<float> Sensors { get; }

        public IReadOnlyList<float> Effectors { get; }
        
        public void Activate()
        {
            Activate(true);
        }

        private void Activate(bool resetPreActivations)
        {
            //bias & sensors
            for (var i = 0; i < _inputCount; i++)
            {
                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _preActivation[connection.TargetIdx] += _postActivation[i] * connection.Weight;
                }
            }

            //hidden & outputs
            for (var i = _connections.Length - 1; i >= _inputCount; i--)
            {
                var preActivation = _preActivation[i];
                var preActivationAbs = preActivation < 0 ? -preActivation : preActivation;
                _postActivation[i] = preActivationAbs > float.Epsilon
                    ? preActivation / (0.5f + preActivationAbs) //inlined zero-centered softsign
                    : 0;

                if (resetPreActivations)
                {
                    _preActivation[i] = 0;
                }

                for (var j = 0; j < _connections[i].Length; j++)
                {
                    var connection = _connections[i][j];
                    _preActivation[connection.TargetIdx] += _postActivation[i] * connection.Weight;
                }
            }
        }

        public void Train(float[][] samples, float learningRate = 0.01f, float l1Ratio = 0, float l2Ratio = 0)
        {
            for (var i = 0; i < samples.Length; i++)
            {
                TrainIncremental(samples[i], learningRate, l1Ratio, l2Ratio);
            }

            _encoder.Encode();
        }

        public void ActivateAndTrain(float[] samples, float learningRate = 0.01f, float l1Ratio = 0, float l2Ratio = 0)
        {
            TrainIncremental(samples, learningRate, l1Ratio, l2Ratio);

            _encoder.Encode();
        }

        private void TrainIncremental(float[] samples, float learningRate, float l1Ratio, float l2Ratio)
        {
            for (var i = 0; i < Sensors.Count; ++i)
            {
                Sensors[i] = samples[i];
            }

            Activate(false);

            PropagateError(new ArraySegment<float>(samples, Sensors.Count, Effectors.Count),
                learningRate,
                l1Ratio, l2Ratio);
        }

        private void PropagateError(IReadOnlyList<float> expectations, float learningRate, float l1Ratio, float l2Ratio)
        {
            // outputs
            PropagateOuterError(expectations);

            // hidden
            for (var i = _inputCount + Effectors.Count; i < _connections.Length; ++i)
                PropagateInnerError(i, learningRate, l1Ratio, l2Ratio);

            // inputs & bias
            for (var i = 0; i < _inputCount; ++i)
                PropagateInputError(i, learningRate, l1Ratio, l2Ratio);
        }

        private void PropagateOuterError(IReadOnlyList<float> expectations)
        {
            for (var j = 0; j < Effectors.Count; ++j)
            {
                _gradient[_inputCount + j] =
                    GetDerivative(_preActivation[_inputCount + j]) * (Effectors[j] - expectations[j]);
                _preActivation[_inputCount + j] = 0;
            }
        }
        
        private void PropagateInnerError(int index, float learningRate, float l1Ratio, float l2Ratio)
        {
            _gradient[index] = 0;
                
            for (var j = 0; j < _connections[index].Length; j++)
            {
                var connection = _connections[index][j];
                _gradient[index] += _gradient[connection.TargetIdx] * connection.Weight;
                _connections[index][j] =
                    GetModifiedConnection(connection, _postActivation[index], learningRate, l1Ratio, l2Ratio);
            }

            _gradient[index] *= GetDerivative(_preActivation[index]);
            _preActivation[index] = 0;
        }
        
        private void PropagateInputError(int index, float learningRate, float l1Ratio, float l2Ratio)
        {
            for (var j = 0; j < _connections[index].Length; j++)
            {
                _connections[index][j] = GetModifiedConnection(_connections[index][j], _postActivation[index],
                    learningRate, l1Ratio, l2Ratio);
            }
        }

        private NetworkConnection GetModifiedConnection(NetworkConnection connection, float input,
            float learningRate, float l1Ratio, float l2Ratio)
        {
            var oldWeight = connection.Weight;
            var newWeight = oldWeight - learningRate * (input * _gradient[connection.TargetIdx] +
                                                        l1Ratio * Math.Sign(oldWeight) +
                                                        l2Ratio * oldWeight);
            return new NetworkConnection(connection.TargetIdx, newWeight);
        }

        private static float GetDerivative(float x)
        {
            var d = 0.5f + (x < 0 ? -x : x);
            return 0.5f / (d * d);
        }
    }
}