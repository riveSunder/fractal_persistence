# Continuous Cellular Automata 

<div align="center">

<!-- 'video for everyone' code snippet from https://camendesign.com/code/video_for_everybody -->
<!-- first try HTML5 playback: if serving as XML, expand `controls` to `controls="controls"` and autoplay likewise -->
<!-- warning: playback does not work on iOS3 if you include the poster attribute! fixed in iOS4.0 -->
<video width="512" height="512" controls>
	<source src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1c_orbium_unicaudatus.mp4" type="video/mp4" /><!-- Safari / iOS video    -->
<!-- <source src="__VIDEO__.OGV" type="video/ogg" /><!-- Firefox / Opera / Chrome10 --> -->
	<!-- fallback to Flash: -->
	<object width="512" height="512" type="application/x-shockwave-flash" data="__FLASH__.SWF">
		<!-- Firefox uses the `data` attribute above, IE/Safari uses the param below -->
		<param name="movie" value="__FLASH__.SWF" />
		<param name="flashvars" value="controlbar=over&amp;image=__image__.png&amp;file=https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1c_orbium_unicaudatus.mp4" />
		<!-- fallback image. note the title field below, put the title of the video there -->
		<img src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1c_thumbnail.png" width="512" height="512" alt="thumbnail of _Orbium_ glider"
		     title="No video playback capabilities, please download the video below" />
	</object>
</video>
<p>	
  <em>Orbium</em> glider in Lenia CA, progressively smaller step size dt. 
  <br>
  <strong>Download Video:</strong>
	<a href="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1c_orbium_unicaudatus.mp4">"MP4"</a>
<!-- Open Format:	<a href="__VIDEO__.OGV">"Ogg"</a> -->
</p>

</div>

### If we want more accurate simulation, all we have to do is add more compute!

{:style="text-align:center;"}
![Lenia update equation](https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/lenia_update.png)

```
# given
# a: cell states
# k: neighborhood kernel
# dt: step size
# grow: growth function
# conv: convolution function
# clip: clipping function (truncates values)

a = clip(a + dt * grow(conv(a,k)), 0., 1.)
```

{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/fractal_persistence/al24_slide_002) -- [Supporting resources](https://rivesunder.github.io/fractal_persistence) -- [Next slide](https://rivesunder.github.io/fractal_persistence/al24_slide_004)

<br>
<div align="center">

<!-- 'video for everyone' code snippet from https://camendesign.com/code/video_for_everybody -->
<!-- first try HTML5 playback: if serving as XML, expand `controls` to `controls="controls"` and autoplay likewise -->
<!-- warning: playback does not work on iOS3 if you include the poster attribute! fixed in iOS4.0 -->
<video width="512" height="512" controls>
	<source src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1b_orbium_unicaudatus.mp4" type="video/mp4" /><!-- Safari / iOS video    -->
<!-- <source src="__VIDEO__.OGV" type="video/ogg" /><!-- Firefox / Opera / Chrome10 --> -->
	<!-- fallback to Flash: -->
	<object width="512" height="512" type="application/x-shockwave-flash" data="__FLASH__.SWF">
		<!-- Firefox uses the `data` attribute above, IE/Safari uses the param below -->
		<param name="movie" value="__FLASH__.SWF" />
		<param name="flashvars" value="controlbar=over&amp;image=__image__.png&amp;file=https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/assets/vid1b_orbium_unicaudatus.mp4" />
		<!-- fallback image. note the title field below, put the title of the video there -->
		<img src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid1b_thumbnail.png" width="512" height="512" alt="thumbnail of _Orbium_ glider"
		     title="No video playback capabilities, please download the video below" />
	</object>
</video>
<p>	
  <em>Orbium</em> glider in Lenia CA, kernel radius of 130 and step size dt of 0.01 is 10x finer than typical kr=13, dt=0.1
  <br>
  <strong>Download Video:</strong>
	<a href="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/assets/vid1b_orbium_unicaudatus.mp4">"MP4"</a>
<!-- Open Format:	<a href="__VIDEO__.OGV">"Ogg"</a> -->
</p>

</div>
