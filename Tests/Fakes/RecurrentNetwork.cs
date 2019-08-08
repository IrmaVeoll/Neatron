using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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
        private readonly Dictionary<int, List<Connection>> _сonnectionsMap = new Dictionary<int, List<Connection>>(); // id to links targeting this neuron
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
            if (_сonnectionsMap.ContainsKey(targetId))
            {
                _сonnectionsMap[targetId].Add(connection);
            }
            else
            {
                _сonnectionsMap[targetId] = new List<Connection>() {connection};
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

            foreach (var neuron in _neurons.Values)
            {
                if (_сonnectionsMap.TryGetValue(neuron.Id, out var connections))
                {
                    foreach (var connection in connections)
                    {
                        neuron.CurrentActivation += connection.Weight *
                                                    _neurons[connection.SourceId].PreviousActivation;
                    }
                }

                if(neuron.Layer > 0)
                    neuron.CurrentActivation = ActivationFunction(neuron.CurrentActivation);          
            }

            return _outputs.Select(n => n.CurrentActivation).ToArray();
        }

        private float ActivationFunction(float x)
        {
            return x / (0.5f + Math.Abs(x));
        }
    }
}