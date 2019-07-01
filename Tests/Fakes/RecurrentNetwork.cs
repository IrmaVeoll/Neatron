using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;

namespace Tests.Fakes
{
    // literally the most simple and ineffective implementation of recurrent neural network
    // use at your own risk
    // better not use at all
    internal class RecurrentNetwork
    {
        private class Neuron
        {
            public Neuron(int id, int layer)
            {
                Debug.Assert(layer >= 0);
                
                Id = id;
                Layer = layer;
                PreviousActivation = 0f;
                CurrentActivation = 0f;
            }

            public readonly int Id;
            public readonly int Layer;
            public float PreviousActivation;
            public float CurrentActivation;
        }

        private struct Connection
        {
            public int SourceId;
            public int TargetId;
            public float Weight;
        }

        private readonly Dictionary<int, Neuron> _neurons = new Dictionary<int, Neuron>();
        private readonly Dictionary<int, List<Connection>> _forwardConnectionMap = new Dictionary<int, List<Connection>>();
        private readonly Dictionary<int, List<Connection>> _backwardConnectionMap = new Dictionary<int, List<Connection>>();
        private bool _isFrozen;
        private Neuron[] _inputs;
        private Neuron[] _outputs;

        public int ConnectionsCount { get; private set; }

        public void AddNeuron(int id, int layer)
        {
            if (_isFrozen)
                throw new InvalidOperationException("No structural changes after activation!");
            
            _neurons[id] = (new Neuron(id, layer));
        }

        public void AddConnection(int sourceId, int targetId, float weight)
        {
            if (_isFrozen)
                throw new InvalidOperationException("No structural changes after activation!");

            ConnectionsCount++;
            
            var connection = new Connection {SourceId = sourceId, TargetId = targetId, Weight = weight};
            var isForward = _neurons[targetId].Layer > _neurons[sourceId].Layer;
            if (isForward)
            {
                if(_forwardConnectionMap.ContainsKey(sourceId))
                    _forwardConnectionMap[sourceId].Add(connection);
                else
                    _forwardConnectionMap[sourceId] = new List<Connection>() {connection};
            }
            else
            {
                if(_backwardConnectionMap.ContainsKey(targetId))
                    _backwardConnectionMap[targetId].Add(connection);
                else
                    _backwardConnectionMap[targetId] = new List<Connection>() {connection};          
            }
        }
        
        public float[] Activate(float[] values)
        {
            if (!_isFrozen)
            {
                _inputs = _neurons.Values.Where(n => n.Layer == 0).OrderBy(n => n.Id).ToArray();
                if(_inputs.Length == 0)
                    throw new InvalidOperationException("No inputs in the network!");
                
                var maxLayer = _neurons.Values.Max(n => n.Layer);
                _outputs = _neurons.Values.Where(n => n.Layer == maxLayer).ToArray();
            }

            _isFrozen = true;
            
            if(values.Length != _inputs.Length)
                throw new InvalidOperationException("Input size mismatch!");

            foreach (var neuron in _neurons.Values)
            {
                neuron.PreviousActivation = neuron.CurrentActivation;
                neuron.CurrentActivation = 0f;
            }

            for(var i = 0; i < values.Length; i++)
            {
                _inputs[i].CurrentActivation = values[i];
            }

            foreach (var neuron in _neurons.Values.OrderBy(n => n.Layer))
            {
                if (_backwardConnectionMap.TryGetValue(neuron.Id, out var connections))
                {
                    foreach (var connection in connections)
                    {
                        neuron.CurrentActivation += connection.Weight *
                                                    _neurons[connection.SourceId].PreviousActivation;
                    }
                }

                if(neuron.Layer > 0)
                    neuron.CurrentActivation = ActivationFunction(neuron.CurrentActivation);
                
                if (_forwardConnectionMap.TryGetValue(neuron.Id, out connections))
                {
                    foreach (var connection in connections)
                    {
                        _neurons[connection.TargetId].CurrentActivation += connection.Weight * neuron.CurrentActivation;
                    }
                }            
            }

            return _outputs.Select(n => n.CurrentActivation).ToArray();
        }

        private float ActivationFunction(float x)
        {
            return x / (0.5f + Math.Abs(x));
        }
    }
}