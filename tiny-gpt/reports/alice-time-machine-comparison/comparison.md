# Alice + The Time Machine: original versus updated

Original step 1000; updated step 3000. Exact vocabulary and architecture match.

Every holdout target except the first character is scored exactly once. Windows are contiguous, non-overlapping, at most 128 inputs; the context resets at each window. Loss is token-weighted cross-entropy in natural-log units. These full-holdout losses differ from the earlier randomly sampled validation metric. Both models use identical windows.

| Book | Validation targets | Original loss | Updated loss | Change |
| --- | --- | --- | --- | --- |
| alice | 14,459 | 1.8485 | 1.4871 | -0.3613 |
| time_machine | 17,973 | 2.1184 | 1.6093 | -0.5091 |

## Identical sampling settings

Both models: device cuda, temperature 0.8, top-k 20, 200 new characters, seeds [42, 1337, 2026]. Prompt text is included in each output. Evaluation mode, no gradients.

Markdown removes trailing spaces from sample lines; `comparison.json` preserves exact output. The nine matched prompt/seed pairs show variations, rather than selecting a single flattering output. Better holdout loss is evidence of better next-character prediction on these books, not coherent storytelling, factual knowledge, or reasoning. Mixing reduces the risk of forgetting; it does not guarantee preservation of prior outputs.

## Prompt "\n"; seed 42

Original:

```text





                                             w                    s                                                          *
* d                                        f           *s    d
```

Updated:

```text

I’m heardles the souch of the rooth man all to cowder, it was the
holdsehons sat of maching? Alice found earned the that in the edgest
woder, and said “Come across, if the blows.”


“Soup-eses as tea
```

## Prompt "\n"; seed 1337

Original:

```text



bit suche shes ind a has lotime of it it a a groct-thing a bast one, I littleme
gor. you kelow it, you do know it you do kid you ory kbee wamy
crint it he seache said, ait fothers they went to the l
```

Updated:

```text

you the Queen such stange, and
something at talking it in a crast of the light monzer. ‘In appeared,
in was a grovour vive it a knook partes. And rink the Morlacte said
hairs how size: “No old, at it
```

## Prompt "\n"; seed 2026

Original:

```text

     she couldn ton the roouse a little thats sionace all imome that
thers begrentinglinf, int formed to look cantere, sthis same she
downd and on ofe they, and the she swintent shume, she dit moom no
```

Updated:

```text


“Terse to it,” the March Mock Turtle tursively, and she was a planget the
blackness litf, and for the villing with a clack, and the story.

For sight to close the sun!”

“He near distured in or of th
```

## Prompt "Alice "; seed 42

Original:

```text
Alice bote has to it the couse.

The Queeme gay,” so that she the Mock Moulrse Alice. “I mouself to mave mine
it, a I din that ine, as do loy yourn’t that cone was clo it if thay
littere a sallle thought th
```

Updated:

```text
Alice both had thought, and he think said the little of days, and the
talkless another the condes? Alice found earn aruing had in a sort old
something. “Grye, you know it is torn had after as she sloperder
```

## Prompt "Alice "; seed 1337

Original:

```text
Alice you the was she wits a has lor
some the gat tas itchirts in that do fit pliat smong raping and the
a sseen it. And you the kit
shomorile wen a that! I knee to a yetsaid, tit forme in to rouply
they he
```

Updated:

```text
Alice or the general sency and but time of the more of the
things heartous for lifted this appearach.

“Walls was argring you down the story, but I made it was he same engager to
tell moving arms the more l
```

## Prompt "Alice "; seed 2026

Original:

```text
Alice the ras, and said not wor to a a little thats sio
as shing mouters ont the gliked bbole farnct hous do on the cane to
entoing at thems of the of she know chang to his the hat ne shume,
she dit moom no
```

Updated:

```text
Alice the right only my now words. The simply, save signacce, and Alice a
horrroally, with litfericty of the only that the went is same shouter
down and the conder smolet sen, with I do never that tormouse
```

## Prompt "The Time Traveller "; seed 42

Original:

```text
The Time Traveller it, and bed it little wey tinke and of anly as a
wous hit, “I lit helrse to mak allkice littlats you to heave a
dince a aind wis if loy odldn’t that conmed tamlokit if thay looke
forte tase thought th
```

Updated:

```text
The Time Traveller box, was that glass camey time sat madd, and
that dayly the meant hillsess.

“Have more lessonation,” said the ruins had in a voice, “He descovers and
no we acto it is than horied the stare of the rig
```

## Prompt "The Time Traveller "; seed 1337

Original:

```text
The Time Traveller one the Quees one, you knoty time of greave tas ince-thi
said a to she welll of thite poon a had it, “s was that in you dof
thatkers,” said Alice ritt the Marcache said had the wening in
tho thatt gai
```

Updated:

```text
The Time Traveller as it such fourst, you know. I
say, I did the world hap him heard only delights on the tun apped it,
as was all brough the Morlocks riseblet. And rink the Morlocks said
hairs how size: “Four you me ha
```

## Prompt "The Time Traveller "; seed 2026

Original:

```text
The Time Traveller they she come a ling to row.

“No it see says __onacce, which fils on by sable, whing thalking formed
tonly that the went it sampe ons orme grouon ofe tonce, “It lonst!”

you no she like was to do bet
```

Updated:

```text
The Time Traveller have shouck once.

“The roof,” said Alice shoutly as she was a planget the blackness diff,
and for the villing with and this same some or the poor of a parm,
I the surpred winkling my usup own muth ha
```
