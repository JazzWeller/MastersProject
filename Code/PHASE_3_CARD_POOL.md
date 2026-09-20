# Phase 3 Card Pool — Call of the Archons: Brobnar, Mars, Sanctum, Untamed

Source: keyteki `packs/CotA.json` for card text/stats, cross-checked against the official Master Vault decks API (`/api/decks/?expansion=<set>&links=cards`) for the two art reprint sets keyteki has no pack for (907, 964; three remaining unresolved images -- Mighty Lance, Kindrith Longshot, Brothers in Battle -- were read directly off the card art, since the name is printed on every card). Errata from the official KeyForge Master Rulebook v18.3 (Ghost Galaxy, Feb 2026). Art files are newer reprints of the same cards; art-to-card mapping is a verified bijection (every image maps to exactly one CotA card in that house and vice versa) -- see `Code/Non-GUI/tools/build_card_data.py` and `tests/test_card_registry.py`.

Canonical text = CotA text + errata. The general “Fight:”/“Reap:” → “After Fight:”/“After Reap:” errata reading (MRB 18.3, General Errata) is applied silently to every card whose printed text uses the bare form; those rows are marked “general Fight/Reap errata” in the Errata column, matching Phase 2's convention. Armor is a new column in this table -- Phase 2's 159-card pool had no armored creatures.

## Brobnar

| # | Card | Type | Pow | Armor | Æ | Traits | Keywords | Canonical text | Errata | Art | P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | Anger | action |  |  | 1 |  |  | Play: Ready and fight with a friendly creature. |  | `Phase 3/Cards/Brobnar/907-037.png` |  |
| 002 | Barehanded | action |  |  | 1 |  |  | Play: Put each artifact on top of its owner’s deck. |  | `Phase 3/Cards/Brobnar/964-001.png` |  |
| 003 | Blood Money | action |  |  |  |  |  | Play: Place 2Æ from the common supply on an enemy creature. |  | `Phase 3/Cards/Brobnar/907-020.png` |  |
| 004 | Brothers in Battle | action |  |  | 1 |  |  | Play: Choose a house. For the remainder of the turn, each friendly creature of that house may fight. |  | `Phase 3/Cards/Brobnar/964-004.png` |  |
| 005 | Burn the Stockpile | action |  |  |  |  |  | Play: If your opponent has 7Æ or more, they lose 4Æ. |  | `Phase 3/Cards/Brobnar/907-022.png` |  |
| 006 | Champion’s Challenge | action |  |  |  |  |  | Play: Destroy each enemy creature except the most powerful enemy creature. Destroy each friendly creature except the most powerful friendly creature. Ready and fight with your remaining creature. |  | `Phase 3/Cards/Brobnar/918-005.png` |  |
| 007 | Coward’s End | action |  |  |  |  |  | Play: Destroy each undamaged creature. Gain 3 chains. |  | `Phase 3/Cards/Brobnar/435-004.png` |  |
| 008 | Follow the Leader | action |  |  |  |  |  | Play: For the remainder of the turn, each friendly creature may fight. |  | `Phase 3/Cards/Brobnar/907-027.png` |  |
| 009 | Lava Ball | action |  |  |  |  |  | Play: Deal 4D to a creature with 2D splash. |  | `Phase 3/Cards/Brobnar/918-011.png` |  |
| 010 | Loot the Bodies | action |  |  |  |  |  | Play: For the remainder of the turn, gain 1Æ each time an enemy creature is destroyed. |  | `Phase 3/Cards/Brobnar/341-010.png` |  |
| 011 | Take that, Smartypants | action |  |  | 1 |  |  | Play: Steal 2Æ if your opponent has 3 or more Logos cards in play. |  | `Phase 3/Cards/Brobnar/435-050.png` |  |
| 012 | Punch | action |  |  | 1 |  |  | Play: Deal 3D to a creature. |  | `Phase 3/Cards/Brobnar/341-012.png` |  |
| 013 | Relentless Assault | action |  |  |  |  |  | Play: Ready and fight with up to 3 different friendly creatures, one at a time. |  | `Phase 3/Cards/Brobnar/964-022.png` |  |
| 014 | Smith | action |  |  | 1 |  |  | Play: Gain 2Æ if you control more creatures than your opponent. |  | `Phase 3/Cards/Brobnar/964-025.png` |  |
| 015 | Sound the Horns | action |  |  | 1 |  |  | Play: Discard cards from the top of your deck until you either discard a Brobnar creature or run out of cards. If you discarded a Brobnar creature this way, put it into your hand. |  | `Phase 3/Cards/Brobnar/722-039.png` |  |
| 016 | Tremor | action |  |  |  |  |  | Play: Stun a creature and each of its neighbors. |  | `Phase 3/Cards/Brobnar/452-015.png` |  |
| 017 | Unguarded Camp | action |  |  | 1 |  |  | Play: For each creature you have in excess of your opponent, a friendly creature captures 1Æ. Each creature cannot capture more than 1Æ this way. |  | `Phase 3/Cards/Brobnar/907-034.png` |  |
| 018 | Warsong | action |  |  |  |  |  | Play: For the remainder of the turn, gain 1Æ each time a friendly creature fights. |  | `Phase 3/Cards/Brobnar/341-018.png` |  |
| 019 | Autocannon | artifact |  |  | 1 | weapon |  | Deal 1D to each creature after it enters play. |  | `Phase 3/Cards/Brobnar/600-002.png` |  |
| 020 | Banner of Battle | artifact |  |  |  | item |  | Each friendly creature gets +1 power. |  | `Phase 3/Cards/Brobnar/600-003.png` |  |
| 021 | Cannon | artifact |  |  |  | weapon |  | Action: Deal 2D to a creature. |  | `Phase 3/Cards/Brobnar/907-023.png` |  |
| 022 | Gauntlet of Command | artifact |  |  |  | item |  | Action: Ready and fight with a friendly creature. |  | `Phase 3/Cards/Brobnar/964-033.png` |  |
| 023 | Iron Obelisk | artifact |  |  |  | location |  | Your opponent’s keys cost +1Æ for each friendly damaged Brobnar creature. |  | `Phase 3/Cards/Brobnar/964-009.png` |  |
| 024 | Mighty Javelin | artifact |  |  | 1 | weapon |  | Omni: Sacrifice Mighty Javelin. Deal 4D to a creature. |  | `Phase 3/Cards/Brobnar/907-030.png` |  |
| 025 | Pile of Skulls | artifact |  |  |  | location |  | Each time an enemy creature is destroyed during your turn, a friendly creature captures 1Æ. |  | `Phase 3/Cards/Brobnar/918-012.png` |  |
| 026 | Screechbomb | artifact |  |  |  | weapon |  | Omni: Sacrifice Screechbomb. Your opponent loses 2Æ. |  | `Phase 3/Cards/Brobnar/964-024.png` |  |
| 027 | The Warchest | artifact |  |  |  | item |  | Action: Gain 1Æ for each enemy creature that was destroyed in a fight this turn. |  | `Phase 3/Cards/Brobnar/700-036.png` |  |
| 028 | Bilgum Avalanche | creature | 5 |  |  | giant |  | After you forge a key, deal 2D to each enemy creature. |  | `Phase 3/Cards/Brobnar/907-003.png` |  |
| 029 | Valdr | creature | 6 |  |  | giant |  | Valdr deals +2D while attacking an enemy creature on the flank. |  | `Phase 3/Cards/Brobnar/918-037.png` |  |
| 030 | Bumpsy | creature | 5 |  |  | giant |  | Play: Your opponent loses 1Æ. |  | `Phase 3/Cards/Brobnar/907-041.png` |  |
| 031 | Earthshaker | creature | 7 |  |  | giant |  | Play: Destroy each creature with power 3 or lower. |  | `Phase 3/Cards/Brobnar/907-025.png` |  |
| 032 | Firespitter | creature | 5 | 1 |  | giant |  | Before Fight: Deal 1D to each enemy creature. |  | `Phase 3/Cards/Brobnar/600-041.png` |  |
| 033 | Ganger Chieftain | creature | 5 |  |  | giant |  | Play: You may ready and fight with a neighboring creature. |  | `Phase 3/Cards/Brobnar/964-032.png` |  |
| 034 | Grenade Snib | creature | 2 |  |  | goblin |  | Destroyed: Your opponent loses 2Æ. |  | `Phase 3/Cards/Brobnar/918-021.png` |  |
| 035 | Headhunter | creature | 5 |  |  | giant |  | After Fight: Gain 1Æ. | general Fight/Reap errata | `Phase 3/Cards/Brobnar/964-034.png` |  |
| 036 | Hebe the Huge | creature | 6 |  |  | giant, knight |  | Play: Deal 2D to each other undamaged creature. |  | `Phase 3/Cards/Brobnar/907-029.png` |  |
| 037 | Kelifi Dragon | creature | 12 |  |  | dragon |  | Kelifi Dragon cannot be played unless you have 7Æ or more.After Fight/After Reap: Gain 1Æ. Deal 5D to a creature. | general Fight/Reap errata | `Phase 3/Cards/Brobnar/800-009.png` |  |
| 038 | King of the Crag | creature | 7 |  |  | giant |  | Each enemy Brobnar creature gets –2 power. |  | `Phase 3/Cards/Brobnar/435-041.png` |  |
| 039 | Krump | creature | 6 |  |  | giant |  | After an enemy creature is destroyed fighting Krump, its controller loses 1Æ. |  | `Phase 3/Cards/Brobnar/341-039.png` |  |
| 040 | Lomir Flamefist | creature | 5 |  |  | giant |  | Play: If your opponent has 7Æ or more, they lose 2Æ. |  | `Phase 3/Cards/Brobnar/341-040.png` |  |
| 041 | Looter Goblin | creature | 2 |  |  | goblin | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)After Reap: For the remainder of the turn, gain 1Æ each time an enemy creature is destroyed. | general Fight/Reap errata | `Phase 3/Cards/Brobnar/907-012.png` |  |
| 042 | Mugwump | creature | 6 |  |  | giant |  | After an enemy creature is destroyed fighting Mugwump, fully heal Mugwump and give it a +1 power counter. |  | `Phase 3/Cards/Brobnar/964-010.png` |  |
| 043 | Pingle Who Annoys | creature | 2 |  |  | goblin | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)Deal 1D to each enemy creature after it enters play. |  | `Phase 3/Cards/Brobnar/964-011.png` |  |
| 044 | Rock-Hurling Giant | creature | 6 |  |  | giant |  | During your turn, each time you discard a Brobnar card from your hand, you may deal 4D to a creature. |  | `Phase 3/Cards/Brobnar/700-016.png` |  |
| 045 | Rogue Ogre | creature | 6 |  |  | giant, mutant |  | At the end of your turn, if you played exactly one card this turn, Rogue Ogre heals 2 damage and captures 1Æ. |  | `Phase 3/Cards/Brobnar/341-045.png` |  |
| 046 | Smaaash | creature | 5 |  |  | giant |  | Play: Stun a creature. |  | `Phase 3/Cards/Brobnar/341-046.png` |  |
| 047 | Tireless Crocag | creature | 7 |  |  | giant |  | Tireless Crocag cannot reap.You may use Tireless Crocag as if it belonged to the active house.If your opponent has no creatures in play, destroy Tireless Crocag. |  | `Phase 3/Cards/Brobnar/907-017.png` |  |
| 048 | Troll | creature | 8 |  |  | giant |  | After Reap: Troll heals 3 damage. | general Fight/Reap errata | `Phase 3/Cards/Brobnar/907-051.png` |  |
| 049 | Wardrummer | creature | 3 |  |  | goblin |  | Play: Return each other friendly Brobnar creature to your hand. |  | `Phase 3/Cards/Brobnar/341-049.png` |  |
| 050 | Blood of Titans | upgrade |  |  | 1 |  |  | This creature gets +5 power. |  | `Phase 3/Cards/Brobnar/907-021.png` |  |
| 051 | Phoenix Heart | upgrade |  |  |  |  |  | This creature gains, “Destroyed: Return this creature to its owner’s hand and deal 3D to each creature in play.” |  | `Phase 3/Cards/Brobnar/435-045.png` |  |
| 052 | Yo Mama Mastery | upgrade |  |  | 1 |  |  | This creature gains taunt.Play: Fully heal this creature. |  | `Phase 3/Cards/Brobnar/964-015.png` |  |

## Mars

| # | Card | Type | Pow | Armor | Æ | Traits | Keywords | Canonical text | Errata | Art | P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 160 | Ammonia Clouds | action |  |  |  |  |  | Play: Deal 3D to each creature. |  | `Phase 3/Cards/Mars/600-151.png` |  |
| 161 | Battle Fleet | action |  |  | 1 |  |  | Play: Reveal any number of Mars cards from your hand. For each card revealed this way, draw 1 card. |  | `Phase 3/Cards/Mars/939-143.png` |  |
| 162 | Deep Probe | action |  |  | 1 |  |  | Play: Choose a house. Reveal your opponent's hand. Discard each creature of that house revealed this way. |  | `Phase 3/Cards/Mars/939-144.png` |  |
| 163 | EMP Blast | action |  |  | 1 |  |  | Play: Each Mars creature and each Robot creature is stunned. Each artifact is destroyed. |  | `Phase 3/Cards/Mars/341-163.png` |  |
| 164 | Hypnotic Command | action |  |  |  |  |  | Play: For each friendly Mars creature, choose an enemy creature to capture 1Æ from their own side. |  | `Phase 3/Cards/Mars/939-131.png` |  |
| 165 | Irradiated Æmber | action |  |  | 1 |  |  | Play: If your opponent has 6Æ or more, deal 3D to each enemy creature. |  | `Phase 3/Cards/Mars/939-133.png` |  |
| 166 | Key Abduction | action |  |  | 1 |  |  | Play: Return each Mars creature to its owner's hand. Then, you may forge a key at +9Æ current cost, reduced by 1Æ for each card in your hand. |  | `Phase 3/Cards/Mars/939-147.png` |  |
| 167 | Martian Hounds | action |  |  |  |  |  | Play: Choose a creature. For each damaged creature, give the chosen creature two +1 power counters. |  | `Phase 3/Cards/Mars/435-203.png` |  |
| 168 | Martians Make Bad Allies | action |  |  |  |  |  | Play: Reveal your hand. Purge each revealed non-Mars creature and gain 1Æ for each card purged this way. |  | `Phase 3/Cards/Mars/939-134.png` |  |
| 169 | Mass Abduction | action |  |  | 1 |  |  | Play: Put up to 3 damaged enemy creatures into your archives. If any of these creatures leave your archives, they are put into their owner’s hand instead. |  | `Phase 3/Cards/Mars/928-090.png` |  |
| 170 | Mating Season | action |  |  | 1 |  |  | Play: Shuffle each Mars creature into its owner’s deck. Each player gains 1Æ for each creature shuffled into their deck this way. |  | `Phase 3/Cards/Mars/918-095.png` |  |
| 171 | Mothership Support | action |  |  | 1 |  |  | Play: For each friendly ready Mars creature, deal 2D to a creature. (You may choose a different creature each time.) |  | `Phase 3/Cards/Mars/600-142.png` |  |
| 172 | Orbital Bombardment | action |  |  | 1 |  |  | Play: Reveal any number of Mars cards from your hand. For each card revealed this way, deal 2D to a creature. (You may choose a different creature each time.) |  | `Phase 3/Cards/Mars/939-151.png` |  |
| 173 | Phosphorus Stars | action |  |  |  |  |  | Play: Stun each non-Mars creature. Gain 2 chains. |  | `Phase 3/Cards/Mars/341-173.png` |  |
| 174 | Psychic Network | action |  |  |  |  |  | Play: Steal 1Æ for each friendly ready Mars creature. |  | `Phase 3/Cards/Mars/700-197.png` |  |
| 175 | Sample Collection | action |  |  |  |  |  | Play: Put an enemy creature into your archives for each key your opponent has forged. If any of these creatures leave your archives, they are put into their owner’s hand instead. |  | `Phase 3/Cards/Mars/435-188.png` |  |
| 176 | Shatter Storm | action |  |  |  |  |  | Play: Lose all your Æ. Then, your opponent loses triple the amount of Æ you lost this way. |  | `Phase 3/Cards/Mars/939-138.png` |  |
| 177 | Soft Landing | action |  |  |  |  |  | Play: The next creature or artifact you play this turn enters play ready. |  | `Phase 3/Cards/Mars/341-177.png` |  |
| 178 | Squawker | action |  |  | 1 |  |  | Play: Ready a Mars creature or stun a non-Mars creature. |  | `Phase 3/Cards/Mars/918-119.png` |  |
| 179 | Total Recall | action |  |  | 1 |  |  | Play: For each friendly ready creature, gain 1Æ. Return each friendly creature to your hand. |  | `Phase 3/Cards/Mars/800-231.png` |  |
| 180 | Combat Pheromones | artifact |  |  | 1 | item |  | Omni: Sacrifice Combat Pheromones. You may use up to 2 other Mars cards this turn. |  | `Phase 3/Cards/Mars/435-177.png` |  |
| 181 | Commpod | artifact |  |  |  | item |  | Action: Reveal any number of Mars cards from your hand. For each card revealed this way, you may ready one Mars creature. |  | `Phase 3/Cards/Mars/939-128.png` |  |
| 182 | Crystal Hive | artifact |  |  |  | location |  | Action: For the remainder of the turn, gain 1Æ each time a creature reaps. |  | `Phase 3/Cards/Mars/918-102.png` |  |
| 183 | Custom Virus | artifact |  |  | 1 | weapon |  | Omni: Destroy Custom Virus. You may purge a creature from your hand. If you do, destroy each creature that shares a trait with the purged creature. | MRB 18.3 errata | `Phase 3/Cards/Mars/918-089.png` |  |
| 184 | Feeding Pit | artifact |  |  |  | location |  | Action: Discard a creature from your hand. If you do, gain 1Æ. |  | `Phase 3/Cards/Mars/700-172.png` |  |
| 185 | Invasion Portal | artifact |  |  |  | location |  | Action: Discard cards from the top of your deck until you discard a Mars creature or run out of cards. If you discard a Mars creature this way, put it into your hand. |  | `Phase 3/Cards/Mars/939-132.png` |  |
| 186 | Incubation Chamber | artifact |  |  |  | location |  | Omni: You may reveal a Mars creature from your hand. If you do, archive it. | MRB 18.3 errata | `Phase 3/Cards/Mars/918-093.png` |  |
| 187 | Mothergun | artifact |  |  |  | weapon |  | Action: Reveal any number of Mars cards from your hand. Deal damage to a creature equal to the number of Mars cards revealed this way. |  | `Phase 3/Cards/Mars/939-159.png` |  |
| 188 | Sniffer | artifact |  |  | 1 | ally |  | Action: For the remainder of the turn, each creature loses elusive. |  | `Phase 3/Cards/Mars/700-183.png` |  |
| 189 | Swap Widget | artifact |  |  |  | item |  | Action: Return a ready friendly Mars creature to your hand. If you do, put a Mars creature with a different name from your hand into play, then ready it. |  | `Phase 3/Cards/Mars/939-139.png` |  |
| 190 | Blypyp | creature | 2 |  |  | martian, scientist |  | After Reap: The next Mars creature you play this turn enters play ready. | general Fight/Reap errata | `Phase 3/Cards/Mars/918-101.png` |  |
| 191 | Chuff Ape | creature | 11 |  |  | beast | taunt | Taunt. (This creature’s neighbors cannot be attacked unless they have taunt.)Chuff Ape enters play stunned.After Fight/After Reap: You may sacrifice another friendly creature. If you do, fully heal Chuff Ape. | general Fight/Reap errata | `Phase 3/Cards/Mars/341-191.png` |  |
| 192 | Ether Spider | creature | 7 |  |  | beast |  | Ether Spider deals no damage when fighting.Each Æ that would be added to your opponent’s pool is captured by Ether Spider instead. |  | `Phase 3/Cards/Mars/939-146.png` |  |
| 193 | Grabber Jammer | creature | 4 | 1 |  | robot |  | Your opponent’s keys cost +1Æ.After Fight/After Reap: Capture 1Æ. | general Fight/Reap errata | `Phase 3/Cards/Mars/700-209.png` |  |
| 194 | Grommid | creature | 10 |  |  | beast |  | You cannot play creatures.  After an enemy creature is destroyed fighting Grommid, your opponent loses 1Æ. |  | `Phase 3/Cards/Mars/435-197.png` |  |
| 195 | “John Smyth” | creature | 2 |  |  | agent, martian | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)After Fight/After Reap: Ready a non-Agent Mars creature. | general Fight/Reap errata | `Phase 3/Cards/Mars/341-195.png` |  |
| 196 | Mindwarper | creature | 2 |  |  | martian, scientist | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)Action: Choose an enemy creature. It captures 1Æ from its own side. |  | `Phase 3/Cards/Mars/435-167.png` |  |
| 197 | Phylyx the Disintegrator | creature | 1 |  |  | martian, soldier | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)Action: Your opponent loses 1Æ for each other friendly Mars creature. |  | `Phase 3/Cards/Mars/939-136.png` |  |
| 198 | Qyxxlyx Plague Master | creature | 3 |  |  | martian, scientist |  | After Fight/After Reap: Deal 3D to each Human creature. This damage cannot be prevented by armor. | general Fight/Reap errata | `Phase 3/Cards/Mars/600-128.png` |  |
| 199 | Tunk | creature | 6 | 1 |  | robot |  | After you play another Mars creature, fully heal Tunk. |  | `Phase 3/Cards/Mars/341-199.png` |  |
| 200 | Ulyq Megamouth | creature | 3 |  |  | martian, scientist |  | After Fight/After Reap: Use a friendly non-Mars creature. | general Fight/Reap errata | `Phase 3/Cards/Mars/939-163.png` |  |
| 201 | Uxlyx the Zookeeper | creature | 2 |  |  | martian, scientist | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)After Reap: Put an enemy creature into your archives. If that creature leaves your archives, it is put into its owner’s hand instead. | general Fight/Reap errata | `Phase 3/Cards/Mars/939-152.png` |  |
| 202 | Vezyma Thinkdrone | creature | 3 |  |  | martian, scientist |  | After Reap: You may archive a friendly creature or artifact from play. | general Fight/Reap errata | `Phase 3/Cards/Mars/435-172.png` |  |
| 203 | Yxili Marauder | creature | 2 |  |  | martian, soldier |  | Yxili Marauder gets +1 power for each Æ on it.Play: Capture 1Æ for each friendly ready Mars creature. |  | `Phase 3/Cards/Mars/918-121.png` |  |
| 204 | Yxilo Bolter | creature | 3 |  |  | martian, soldier |  | After Fight/After Reap: Deal 2D to a creature. If this damage destroys that creature, purge it. | general Fight/Reap errata | `Phase 3/Cards/Mars/939-165.png` |  |
| 205 | Yxilx Dominator | creature | 9 | 1 |  | robot | taunt | Taunt. (This creature’s neighbors cannot be attacked unless they have taunt.)Yxilx Dominator enters play stunned. |  | `Phase 3/Cards/Mars/939-166.png` |  |
| 206 | Zorg | creature | 7 |  |  | beast |  | Zorg enters play stunned. Before Fight: Stun the creature Zorg fights and each of that creature’s neighbors. |  | `Phase 3/Cards/Mars/918-111.png` |  |
| 207 | Zyzzix the Many | creature | 3 |  |  | martian, soldier |  | After Fight/After Reap: You may reveal a creature from your hand. If you do, archive it and Zyzzix the Many gets three +1 power counters. | general Fight/Reap errata | `Phase 3/Cards/Mars/341-207.png` |  |
| 208 | Biomatrix Backup | upgrade |  |  | 1 |  |  | This creature gains, “Destroyed: Put this creature into its owner’s archives.” | MRB 18.3 errata | `Phase 3/Cards/Mars/341-208.png` |  |
| 209 | Brain Stem Antenna | upgrade |  |  |  |  |  | This creature gains, “After you play a Mars creature, ready this creature and for the remainder of the turn it belongs to house Mars.” |  | `Phase 3/Cards/Mars/939-127.png` |  |
| 210 | Jammer Pack | upgrade |  |  | 1 |  |  | This creature gains, “Your opponent's keys cost +2Æ.“ |  | `Phase 3/Cards/Mars/700-191.png` |  |
| 211 | Red Planet Ray Gun | upgrade |  |  | 1 |  |  | This creature gains, “After Reap: Choose a creature. Deal 1D to that creature for each Mars creature in play.” | general Fight/Reap errata | `Phase 3/Cards/Mars/600-146.png` |  |

## Sanctum

| # | Card | Type | Pow | Armor | Æ | Traits | Keywords | Canonical text | Errata | Art | P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 212 | Begone! | action |  |  |  |  |  | Play: Choose one: destroy each Dis creature, or gain 1Æ. |  | `Phase 3/Cards/Sanctum/918-128.png` |  |
| 213 | Blinding Light | action |  |  | 1 |  |  | Play: Choose a house. Stun each creature of that house. |  | `Phase 3/Cards/Sanctum/435-214.png` |  |
| 214 | Charge! | action |  |  | 1 |  |  | Play: For the remainder of the turn, each creature you play gains, “Play: Deal 2D to an enemy creature.” |  | `Phase 3/Cards/Sanctum/907-170.png` |  |
| 215 | Cleansing Wave | action |  |  |  |  |  | Play: Heal 1 damage from each creature. Gain 1Æ for each creature healed this way. |  | `Phase 3/Cards/Sanctum/907-206.png` |  |
| 216 | Clear Mind | action |  |  | 1 |  |  | Play: Unstun each friendly creature. |  | `Phase 3/Cards/Sanctum/341-216.png` |  |
| 217 | Doorstep to Heaven | action |  |  | 1 |  |  | Play: Each player with 6Æ or more is reduced to 5Æ. |  | `Phase 3/Cards/Sanctum/918-145.png` |  |
| 218 | Glorious Few | action |  |  |  |  |  | Play: For each creature your opponent controls in excess of you, gain 1Æ. |  | `Phase 3/Cards/Sanctum/907-193.png` |  |
| 219 | Honorable Claim | action |  |  | 1 |  |  | Play: Each friendly Knight creature captures 1Æ. |  | `Phase 3/Cards/Sanctum/918-133.png` |  |
| 220 | Inspiration | action |  |  |  |  |  | Play: Ready and use a friendly creature. |  | `Phase 3/Cards/Sanctum/918-159.png` |  |
| 221 | Mighty Lance | action |  |  |  |  |  | Play: Deal 3D to a creature and 3D to a neighbor of that creature. |  | `Phase 3/Cards/Sanctum/907-177.png` |  |
| 222 | Oath of Poverty | action |  |  | 1 |  |  | Play: Destroy each of your artifacts. Gain 2Æ for each artifact destroyed this way. |  | `Phase 3/Cards/Sanctum/907-180.png` |  |
| 223 | One Stood Against Many | action |  |  | 1 |  |  | Play: Ready and fight with a friendly creature 3 times, each time against a different enemy creature. Resolve these fights one at a time. |  | `Phase 3/Cards/Sanctum/907-181.png` |  |
| 224 | Radiant Truth | action |  |  | 1 |  |  | Play: Stun each enemy creature not on a flank. |  | `Phase 3/Cards/Sanctum/918-151.png` |  |
| 225 | Shield of Justice | action |  |  | 1 |  |  | Play: For the remainder of the turn, each friendly creature cannot be dealt damage. |  | `Phase 3/Cards/Sanctum/918-163.png` |  |
| 226 | Take Hostages | action |  |  | 1 |  |  | Play: For the remainder of the turn, each time a friendly creature fights, it captures 1Æ. |  | `Phase 3/Cards/Sanctum/435-225.png` |  |
| 227 | Terms of Redress | action |  |  | 1 |  |  | Play: Choose a friendly creature to capture 2Æ. |  | `Phase 3/Cards/Sanctum/907-216.png` |  |
| 228 | The Harder They Come | action |  |  |  |  |  | Play: Purge a creature with power 5 or higher. |  | `Phase 3/Cards/Sanctum/907-200.png` |  |
| 229 | The Spirit’s Way | action |  |  |  |  |  | Play: Destroy each creature with power 3 or higher. |  | `Phase 3/Cards/Sanctum/907-201.png` |  |
| 230 | Virtuous Works | action |  |  | 3 |  |  | (Vanilla) |  | `Phase 3/Cards/Sanctum/907-202.png` |  |
| 231 | Epic Quest | artifact |  |  |  | quest |  | Play: Archive each friendly Knight creature in play.Omni: If you have played 7 or more Sanctum cards this turn, sacrifice Epic Quest and forge a key at no cost. |  | `Phase 3/Cards/Sanctum/918-129.png` |  |
| 232 | Gorm of Omm | artifact |  |  |  | item |  | Omni: Sacrifice Gorm of Omm. Destroy an artifact. |  | `Phase 3/Cards/Sanctum/918-146.png` |  |
| 233 | Hallowed Blaster | artifact |  |  |  | weapon |  | Action: Heal 3 damage from a creature. |  | `Phase 3/Cards/Sanctum/341-233.png` |  |
| 234 | Potion of Invulnerability | artifact |  |  | 1 | item |  | Omni: Sacrifice Potion of Invulnerability. For the remainder of the turn, each friendly creature cannot be dealt damage. |  | `Phase 3/Cards/Sanctum/874-164.png` |  |
| 235 | Round Table | artifact |  |  | 1 | location |  | Each friendly Knight creature gets +1 power and gains taunt. |  | `Phase 3/Cards/Sanctum/918-141.png` |  |
| 236 | Sigil of Brotherhood | artifact |  |  | 1 | power |  | Omni: Sacrifice Sigil of Brotherhood. For the remainder of the turn, you may use friendly Sanctum creatures. |  | `Phase 3/Cards/Sanctum/918-154.png` |  |
| 237 | Whispering Reliquary | artifact |  |  |  | item |  | Action: Return an artifact to its owner's hand. |  | `Phase 3/Cards/Sanctum/341-237.png` |  |
| 238 | Bulwark | creature | 4 | 2 |  | human, knight |  | Each of Bulwark’s neighbors gets +2 armor. |  | `Phase 3/Cards/Sanctum/341-238.png` |  |
| 239 | Champion Anaphiel | creature | 6 | 1 |  | knight, spirit | taunt | Taunt. (This creature’s neighbors cannot be attacked unless they have taunt.) |  | `Phase 3/Cards/Sanctum/907-205.png` |  |
| 240 | Champion Tabris | creature | 6 | 2 |  | human, knight |  | After Fight: Capture 1Æ. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/907-189.png` |  |
| 241 | Commander Remiel | creature | 3 |  |  | human, knight |  | After Reap: Use a friendly non-Sanctum creature. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/341-241.png` |  |
| 242 | Duma the Martyr | creature | 3 |  |  | human |  | Destroyed: Fully heal each other friendly creature and draw 2 cards. |  | `Phase 3/Cards/Sanctum/907-171.png` |  |
| 243 | Francus | creature | 6 | 1 |  | knight, spirit |  | After an enemy creature is destroyed fighting Francus, Francus captures 1Æ. |  | `Phase 3/Cards/Sanctum/341-243.png` |  |
| 244 | Grey Monk | creature | 3 |  |  | human, priest |  | Each friendly creature gets +1 armor. After Reap: Heal 2 damage from a creature. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/341-244.png` |  |
| 245 | Hayyel the Merchant | creature | 3 |  |  | human, merchant |  | Each time you play an artifact, gain 1Æ. |  | `Phase 3/Cards/Sanctum/341-245.png` |  |
| 246 | Horseman of Death | creature | 5 |  |  | horseman, spirit |  | Play: Return each Horseman creature from your discard pile to your hand. |  | `Phase 3/Cards/Sanctum/918-167.png` |  |
| 247 | Horseman of Famine | creature | 5 |  |  | horseman, spirit |  | Play/After Fight/After Reap: Destroy the least powerful creature. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/918-168.png` |  |
| 248 | Horseman of Pestilence | creature | 5 |  |  | horseman, spirit |  | Play/After Fight/After Reap: Deal 1D to each non-Horseman creature. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/918-134.png` |  |
| 249 | Horseman of War | creature | 5 |  |  | horseman, spirit |  | Play: For the remainder of the turn, each friendly creature can be used as if they were in the active house, but can only fight. |  | `Phase 3/Cards/Sanctum/918-169.png` |  |
| 250 | Jehu the Bureaucrat | creature | 3 |  |  | human |  | After you choose Sanctum as your active house, gain 2Æ. |  | `Phase 3/Cards/Sanctum/918-135.png` |  |
| 251 | Lady Maxena | creature | 5 | 2 |  | knight, spirit |  | Play: Stun a creature. Action: Return Lady Maxena to its owner’s hand. |  | `Phase 3/Cards/Sanctum/341-251.png` |  |
| 252 | Lord Golgotha | creature | 5 | 2 |  | knight, spirit |  | Before Fight: Deal 3D to each neighbor of the creature Lord Golgotha fights. |  | `Phase 3/Cards/Sanctum/907-175.png` |  |
| 253 | Numquid the Fair | creature | 3 |  |  | human |  | Play: Destroy an enemy creature. Repeat this card’s effect if your opponent still controls more creatures than you. |  | `Phase 3/Cards/Sanctum/907-179.png` |  |
| 254 | Protectrix | creature | 5 |  |  | knight, spirit |  | After Reap: You may fully heal a creature. If you do, that creature cannot be dealt damage for the remainder of the turn. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/341-254.png` |  |
| 255 | Raiding Knight | creature | 4 | 2 |  | human, knight |  | Play: Capture 1Æ. |  | `Phase 3/Cards/Sanctum/918-161.png` |  |
| 256 | Sanctum Guardian | creature | 6 | 1 |  | knight, spirit | taunt | Taunt. (This creature’s neighbors cannot be attacked unless they have taunt.)After Fight/After Reap: Swap Sanctum Guardian with another friendly creature in your battleline. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/918-142.png` |  |
| 257 | Sequis | creature | 4 | 2 |  | human, knight |  | After Reap: Capture 1Æ. | general Fight/Reap errata | `Phase 3/Cards/Sanctum/341-257.png` |  |
| 258 | Sergeant Zakiel | creature | 4 | 1 |  | human, knight |  | Play: You may ready and fight with a neighboring creature. |  | `Phase 3/Cards/Sanctum/341-258.png` |  |
| 259 | Staunch Knight | creature | 4 | 2 |  | human, knight |  | Staunch Knight gets +2 power while it is on a flank. |  | `Phase 3/Cards/Sanctum/496-120.png` |  |
| 260 | Gatekeeper | creature | 5 | 1 |  | knight, spirit |  | Play: If your opponent has 7 or more Æ, capture all but 5 of it. |  | `Phase 3/Cards/Sanctum/907-192.png` |  |
| 261 | The Vaultkeeper | creature | 4 | 1 |  | knight, spirit |  | Your Æ cannot be stolen. |  | `Phase 3/Cards/Sanctum/886-144.png` |  |
| 262 | Veemos Lightbringer | creature | 6 |  |  | angel, spirit |  | Play: Destroy each elusive creature. |  | `Phase 3/Cards/Sanctum/907-186.png` |  |
| 263 | Armageddon Cloak | upgrade |  |  | 1 |  |  | This creature gains hazardous 2 and, “Destroyed: Fully heal this creature and destroy Armageddon Cloak instead.” |  | `Phase 3/Cards/Sanctum/874-133.png` |  |
| 264 | Mantle of the Zealot | upgrade |  |  |  |  |  | This creature gains, “You may use this creature as if it belonged to the active house.” |  | `Phase 3/Cards/Sanctum/886-138.png` |  |
| 265 | Protect the Weak | upgrade |  |  | 1 |  |  | This creature gets +1 armor and gains taunt. (This creature’s neighbors cannot be attacked unless they have taunt.) |  | `Phase 3/Cards/Sanctum/435-221.png` |  |
| 266 | Shoulder Armor | upgrade |  |  | 1 |  |  | While this creature is on a flank, it gets +2 armor and +2 power. |  | `Phase 3/Cards/Sanctum/918-153.png` |  |

## Untamed

| # | Card | Type | Pow | Armor | Æ | Traits | Keywords | Canonical text | Errata | Art | P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 319 | Cooperative Hunting | action |  |  |  |  |  | Play: Deal 1D for each friendly creature in play. You may divide this damage among any number of creatures. |  | `Phase 3/Cards/Untamed/855-306.png` |  |
| 320 | Curiosity | action |  |  | 1 |  |  | Play: Destroy each Scientist creature. |  | `Phase 3/Cards/Untamed/452-387.png` |  |
| 321 | Fertility Chant | action |  |  | 4 |  |  | Play: Your opponent gains 2Æ. |  | `Phase 3/Cards/Untamed/907-334.png` |  |
| 322 | Fogbank | action |  |  | 1 |  |  | Play: Your opponent cannot use creatures to fight on their next turn. |  | `Phase 3/Cards/Untamed/855-297.png` |  |
| 323 | Full Moon | action |  |  |  |  |  | Play: For the remainder of the turn, gain 1Æ each time you play a creature. |  | `Phase 3/Cards/Untamed/964-283.png` |  |
| 324 | Grasping Vines | action |  |  | 1 |  |  | Play: Return up to 3 artifacts to their owners’ hands. |  | `Phase 3/Cards/Untamed/907-353.png` |  |
| 325 | Key Charge | action |  |  |  |  |  | Play: Lose 1Æ. If you do, you may forge a key at current cost. |  | `Phase 3/Cards/Untamed/452-359.png` |  |
| 326 | Lifeweb | action |  |  | 1 |  |  | Play: If your opponent played 3 or more creatures on their previous turn, steal 2Æ. |  | `Phase 3/Cards/Untamed/964-273.png` |  |
| 327 | Lost in the Woods | action |  |  | 1 |  |  | Play: Choose 2 friendly creatures and 2 enemy creatures. Shuffle each chosen creature into its owner’s deck. |  | `Phase 3/Cards/Untamed/907-373.png` |  |
| 328 | Mimicry | action |  |  |  |  |  | When you play this card, treat it as a copy of an action card in your opponent’s discard pile. |  | `Phase 3/Cards/Untamed/700-363.png` |  |
| 329 | Nature’s Call | action |  |  | 1 |  |  | Play: Return up to 3 creatures to their owners’ hands. |  | `Phase 3/Cards/Untamed/964-276.png` |  |
| 330 | Nocturnal Maneuver | action |  |  | 1 |  |  | Play: Exhaust up to 3 creatures. |  | `Phase 3/Cards/Untamed/907-376.png` |  |
| 331 | Perilous Wild | action |  |  | 1 |  |  | Play: Destroy each elusive creature. |  | `Phase 3/Cards/Untamed/964-261.png` |  |
| 332 | Regrowth | action |  |  | 1 |  |  | Play: Return a creature from your discard pile to your hand. |  | `Phase 3/Cards/Untamed/700-381.png` |  |
| 333 | Save the Pack | action |  |  |  |  |  | Play: Destroy each damaged creature. Gain 1 chain. |  | `Phase 3/Cards/Untamed/874-431.png` |  |
| 334 | Scout | action |  |  | 1 |  |  | Play: For the remainder of the turn, up to 2 friendly creatures gain skirmish. Then, fight with those creatures one at a time. |  | `Phase 3/Cards/Untamed/964-264.png` |  |
| 335 | Stampede | action |  |  | 1 |  |  | Play: If you used 3 or more creatures this turn, steal 2Æ. |  | `Phase 3/Cards/Untamed/964-265.png` |  |
| 336 | The Common Cold | action |  |  | 1 |  |  | Play: Deal 1D to each creature. You may destroy all Mars creatures. |  | `Phase 3/Cards/Untamed/918-268.png` |  |
| 337 | Troop Call | action |  |  | 1 |  |  | Play: Return each friendly Niffle creature from your discard pile and from play to your hand. |  | `Phase 3/Cards/Untamed/874-411.png` |  |
| 338 | Vigor | action |  |  | 1 |  |  | Play: Heal up to 3 damage from a creature. If you healed 3 damage, gain 1Æ. |  | `Phase 3/Cards/Untamed/341-338.png` |  |
| 339 | Word of Returning | action |  |  | 1 |  |  | Play: Deal 1D to each enemy creature for each Æ on it. Return all Æ from those creatures to your pool. |  | `Phase 3/Cards/Untamed/964-267.png` |  |
| 340 | Bear Flute | artifact |  |  |  | item |  | Action: Fully heal an Ancient Bear. If there are no Ancient Bears in play, search your deck and discard pile and put each Ancient Bear from them into your hand. If you do, shuffle your discard pile into your deck. |  | `Phase 3/Cards/Untamed/341-340.png` |  |
| 341 | Nepenthe Seed | artifact |  |  |  | item |  | Omni: Sacrifice Nepenthe Seed. Return a card from your discard pile to your hand. |  | `Phase 3/Cards/Untamed/964-277.png` |  |
| 342 | Ritual of Balance | artifact |  |  |  | power |  | Action: If your opponent has 6Æ or more, steal 1Æ. |  | `Phase 3/Cards/Untamed/907-359.png` |  |
| 343 | Ritual of the Hunt | artifact |  |  | 1 | power |  | Omni: Sacrifice Ritual of the Hunt. For the remainder of the turn, you may use friendly Untamed creatures. |  | `Phase 3/Cards/Untamed/918-265.png` |  |
| 344 | World Tree | artifact |  |  |  | location |  | Action: Return a creature from your discard pile to the top of your deck. |  | `Phase 3/Cards/Untamed/964-268.png` |  |
| 345 | Ancient Bear | creature | 5 |  |  | beast | assault:2 | Assault 2.(Before this creature attacks, deal 2D to the attacked enemy.) |  | `Phase 3/Cards/Untamed/341-345.png` |  |
| 346 | Bigtwig | creature | 7 |  |  | beast |  | Bigtwig can only fight stunned creatures. After Reap: Stun and exhaust a creature. | general Fight/Reap errata | `Phase 3/Cards/Untamed/341-346.png` |  |
| 347 | Witch of the Wilds | creature | 4 |  |  | beast, witch |  | During each turn in which Untamed is not your active house, you may play one Untamed card. |  | `Phase 3/Cards/Untamed/918-270.png` |  |
| 348 | Briar Grubbling | creature | 2 |  |  | beast, insect | hazardous:5 | Hazardous 5. (Before this creature is attacked, deal 5D to the attacking enemy.) |  | `Phase 3/Cards/Untamed/855-280.png` |  |
| 349 | Chota Hazri | creature | 3 |  |  | human, witch |  | Play: Lose 1Æ. If you do, you may forge a key at current cost. |  | `Phase 3/Cards/Untamed/435-338.png` |  |
| 350 | Dew Faerie | creature | 2 |  |  | faerie | elusive | Elusive. (The first time this creature is attacked each turn, no damage is dealt.)After Reap: Gain 1Æ. | general Fight/Reap errata | `Phase 3/Cards/Untamed/700-374.png` |  |
| 351 | Dust Pixie | creature | 1 |  | 2 | faerie |  | (Vanilla) |  | `Phase 3/Cards/Untamed/341-351.png` |  |
| 352 | Flaxia | creature | 4 |  |  | faerie |  | Play: Gain 2Æ if you control more creatures than your opponent. |  | `Phase 3/Cards/Untamed/964-271.png` |  |
| 353 | Fuzzy Gruen | creature | 5 |  | 2 | beast |  | Play: Your opponent gains 1Æ. |  | `Phase 3/Cards/Untamed/964-258.png` |  |
| 354 | Giant Sloth | creature | 6 |  |  | beast |  | You cannot use this card unless you have discarded an Untamed card from your hand this turn.Action: Gain 3Æ. |  | `Phase 3/Cards/Untamed/700-339.png` |  |
| 355 | Halacor | creature | 4 |  |  | beast |  | Each friendly flank creature gains skirmish. (When you use a creature with skirmish to fight, it is dealt no damage in return.) |  | `Phase 3/Cards/Untamed/964-272.png` |  |
| 356 | Inka the Spider | creature | 1 |  |  | beast | poison | Poison. (Any damage dealt by this creature’s power during a fight destroys the damaged creature.)Play/After Reap: Stun a creature. | general Fight/Reap errata | `Phase 3/Cards/Untamed/452-392.png` |  |
| 357 | Kindrith Longshot | creature | 3 |  |  | human, ranger | elusive, skirmish | Elusive. Skirmish.After Reap: Deal 2D to a creature. | general Fight/Reap errata | `Phase 3/Cards/Untamed/907-336.png` |  |
| 358 | Snufflegator | creature | 4 |  |  | beast | skirmish | Skirmish. (When you use this creature to fight, it is dealt no damage in return.) |  | `Phase 3/Cards/Untamed/907-380.png` |  |
| 359 | Lupo the Scarred | creature | 6 |  |  | beast | skirmish | Skirmish. (When you use this creature to fight, it is dealt no damage in return.) Play: Deal 2D to an enemy creature. |  | `Phase 3/Cards/Untamed/907-337.png` |  |
| 360 | Mighty Tiger | creature | 4 |  |  | beast |  | Play: Deal 4D to an enemy creature. |  | `Phase 3/Cards/Untamed/907-338.png` |  |
| 361 | Murmook | creature | 3 |  |  | beast |  | Your opponent’s keys cost +1Æ. |  | `Phase 3/Cards/Untamed/918-288.png` |  |
| 362 | Mushroom Man | creature | 2 |  |  | fungus, human |  | Mushroom Man gets +3 power for each unforged key you have. |  | `Phase 3/Cards/Untamed/886-317.png` |  |
| 363 | Niffle Ape | creature | 3 |  |  | beast, niffle |  | While Niffle Ape is attacking, ignore taunt and elusive. |  | `Phase 3/Cards/Untamed/874-424.png` |  |
| 364 | Niffle Queen | creature | 6 |  |  | beast, niffle |  | Each other friendly Beast creature gets +1 power.Each other friendly Niffle creature gets +1 power. |  | `Phase 3/Cards/Untamed/874-409.png` |  |
| 365 | Piranha Monkeys | creature | 2 |  |  | beast |  | Play/After Reap: Deal 2D to each other creature. | general Fight/Reap errata | `Phase 3/Cards/Untamed/907-341.png` |  |
| 366 | Teliga | creature | 3 |  |  | human, witch |  | Each time your opponent plays a creature, gain 1Æ. |  | `Phase 3/Cards/Untamed/964-266.png` |  |
| 367 | Hunting Witch | creature | 2 |  |  | human, witch |  | Each time you play another creature, gain 1Æ. |  | `Phase 3/Cards/Untamed/341-367.png` |  |
| 368 | Witch of the Eye | creature | 3 |  |  | human, witch |  | After Reap: Return a card from your discard pile to your hand. | general Fight/Reap errata | `Phase 3/Cards/Untamed/907-382.png` |  |
| 369 | Way of the Bear | upgrade |  |  | 1 |  |  | This creature gains assault 2. (Before this creature attacks, deal 2D to the attacked enemy.) |  | `Phase 3/Cards/Untamed/341-369.png` |  |
| 370 | Way of the Wolf | upgrade |  |  | 1 |  |  | This creature gains skirmish.  (When you use this creature to fight, it is dealt no damage in return.) |  | `Phase 3/Cards/Untamed/907-364.png` |  |

