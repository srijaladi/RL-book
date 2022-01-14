$V^{\pi D}(s) = Q^{\pi D}(s, \pi_{D}(s))$

$V^{\pi D}(s) = R(s, \pi_{D}(s)) + \gamma \cdot \sum_{s' \in S}^{} P(s, \pi_{D}(s),s') \cdot V^{\pi D}(s')$

$Q^{\pi D}(s, \pi_{D}(s)) = R(s, \pi_{D}(s)) + \gamma \cdot \sum_{s' \in S}^{} P(s, \pi_{D}(s),s') \cdot V^{\pi D}(s')$

$Q^{\pi D}(s, \pi_{D}(s)) = R(s, \pi_{D}(s)) + \gamma \cdot \sum_{s' \in S}^{} P(s, \pi_{D}(s),s') \cdot Q^{\pi D}(s', \pi_{D}(s'))$