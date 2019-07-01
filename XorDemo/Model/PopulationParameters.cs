namespace XorDemo.Model
{
    public readonly struct PopulationParameters
    {
        public readonly int PopulationSize;
        public readonly int TournamentSize;

        public PopulationParameters(int populationSize, int tournamentSize)
        {
            PopulationSize = populationSize;
            TournamentSize = tournamentSize;
        }
    }
}