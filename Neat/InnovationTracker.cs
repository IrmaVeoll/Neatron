using System;
using System.Collections.Generic;

namespace Neat
{
    public sealed class InnovationTracker
    {
        private readonly IDictionary<ConnectionNeurons, int> _addConnectionInnovations =
            new Dictionary<ConnectionNeurons, int>();
        private readonly IDictionary<int, int> _splitConnectionInnovations = new Dictionary<int, int>();
        
        private int _nextNeuronId;
        private int _nextConnectionId;
        
        private readonly object _syncRoot = new object();

        internal InnovationTracker(int firstHiddenNeuronId)
        {
            if (firstHiddenNeuronId <= 0) throw new ArgumentOutOfRangeException(nameof(firstHiddenNeuronId));
            _nextNeuronId = firstHiddenNeuronId;
        }

        public void Reset()
        {
            lock (_syncRoot)
            {
                _addConnectionInnovations.Clear();
                _splitConnectionInnovations.Clear();    
            }
        }

        internal int RegisterAddConnection(ConnectionNeurons connectionNeurons)
        {
            lock (_syncRoot)
            {
                return RegisterInnovation(_addConnectionInnovations,
                    connectionNeurons, 
                    () => _nextConnectionId++);   
            }
        }

        internal int RegisterSplitConnection(in ConnectionGene connectionGene)
        {
            lock (_syncRoot)
            {
                if (_addConnectionInnovations.TryGetValue(connectionGene.ConnectionNeurons, out var id) &&
                    connectionGene.Id == id)
                {
                    _addConnectionInnovations.Remove(connectionGene.ConnectionNeurons);
                }

                return RegisterInnovation(_splitConnectionInnovations, connectionGene.Id, () => _nextNeuronId++);
            }
        }

        private static int RegisterInnovation<T>(IDictionary<T, int> innovations, 
            T innovationKey,
            Func<int> getInnovationId)
        {
            if (!innovations.ContainsKey(innovationKey))
            {
                innovations[innovationKey] = getInnovationId();
            }
            
            return innovations[innovationKey];
        }
    }
}