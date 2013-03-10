The Python audio processing suite
=================================

| Donate to support this free software |
|:------------------------------------:|
| <img width="164" height="164" title="" alt="" src="doc/bitcoin.png" /> |
| [1NttJ7op1xxnM5Gg4qd61Gm7y7yxLbG3F5](bitcoin:1NttJ7op1xxnM5Gg4qd61Gm7y7yxLbG3F5) |

This software contains Python modules and command-line tools with a variety of convenience functions to process and visualize audio signals.  For starters, you can easily plot the spectrum of a song, to discriminate among different-quality versions of the same file.  But the real goal of the suite is to automatically identify duplicates.

Its main feature is the [ButterScotch Butterscotch signature] generator,  which generates a Butterscotch signature out of an audio file.  [ButterScotch Butterscotch signatures] are intended to identify duplicate songs regardless of encoding bitrate, which album or compilation they were released on, start time shifts or song incompleteness.  In addition, given an ideal quantization of the fingerprint, they can reliably serve as unique song identifiers. [ButterScotch Go see why].

The instrumental goal of these algorithms is to robustly identify the same songs released in different albums (but not different takes of said song).  The end goal is to unify the collection statistics of the [http://amarok.kde.org/ Amarok] music player, so that when a song's rating or value changes, that change is propagated to copies of that same song in different albums.  Enterprising users will also be able to use this software to weed out poor-sounding duplicates from their collection -- perhaps even in an automated fashion -- in the future.

You can find the latest news and versions of this software at:
  http://rudd-o.com/new-projects/python-audioprocessing

== What it requires ==

Your computer must have these software packages:

 1. Python
 2. SciPy
 3. NumPy
 4. Matplotlib

All major distributions include these packages, either as default or
in their package manager repositories.  You can get these things in
a matter of minutes, so there's no excuse not to use this software.

== How to get and install it ==

We've moved to https://github.com/Rudd-O/python-audioprocessing

== Help! ==

Sure, we can use the help!  In the above Web site:

 * If you find a bug, please file it using the New Ticket button.
 * If you have patches to contribute, please file a bug too (since we're using Git, chances are we'll be able to pull and push code right from the start).
 * In fact, if you have any idea that you'd like to share with us, do so as well.  Add your proposal to a Proposals page!
