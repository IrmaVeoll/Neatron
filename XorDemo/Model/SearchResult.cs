using System.Collections.Generic;

namespace XorDemo.Model
{
    public readonly struct SearchResult
    {
        public readonly IReadOnlyList<ParetoFrontPoint> FitnessRating;

        public readonly IReadOnlyList<ParetoFrontPoint> SimplicityRating;

        internal SearchResult(
            IReadOnlyList<ParetoFrontPoint> fitnessRating,
            IReadOnlyList<ParetoFrontPoint> simplicityRating)
        {
            FitnessRating = fitnessRating;
            SimplicityRating = simplicityRating;
        }
    }
}
