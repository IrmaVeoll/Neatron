using System.Collections.Generic;
using System.Linq;

namespace Neat
{
    public readonly struct NetworkTopology
    {
        internal NetworkTopology(
            int inputsCount, int outputsCount,
            IReadOnlyList<int> hiddenLayerBounds,
            IReadOnlyList<IReadOnlyList<NetworkConnection>> links)
        {
            LayerRanges = CreateRanges(inputsCount, outputsCount, hiddenLayerBounds);
            Links = links;
            IsValid = true;
        }

        public readonly IReadOnlyList<(int, int)> LayerRanges;
        
        public readonly IReadOnlyList<IReadOnlyList<NetworkConnection>> Links;
        
        public readonly bool IsValid;

        private static (int, int)[] CreateRanges(int inputsCount, int outputsCount, IReadOnlyList<int> hiddenLayerCapacities)
        {
            var result = new (int, int)[hiddenLayerCapacities.Count + 2];
            result[0] = (0, inputsCount);

            var end = inputsCount + outputsCount + hiddenLayerCapacities.Sum();
            for (var i = 1; i <= hiddenLayerCapacities.Count; ++i)
            {
                var capacity = hiddenLayerCapacities[hiddenLayerCapacities.Count - i];
                end -= capacity;
                result[i] = (end, capacity);
            }

            result[result.Length - 1] = (inputsCount, outputsCount);
            return result;
        }
    }
}
