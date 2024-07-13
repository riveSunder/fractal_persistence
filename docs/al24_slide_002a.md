# Continuous Cellular Automata 

<div align="center">

  <!-- 'video for everyone' code snippet from https://camendesign.com/code/video_for_everybody -->
  <!-- first try HTML5 playback: if serving as XML, expand `controls` to `controls="controls"` and autoplay likewise -->
  <!-- warning: playback does not work on iOS3 if you include the poster attribute! fixed in iOS4.0 -->
  <video width="512" height="512" controls>
    <!-- MP4 must be first for iPad! -->
    <source src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid0a_orbium_unicaudatus.mp4" type="video/mp4" /><!-- Safari / iOS video    -->
  <!-- <source src="__VIDEO__.OGV" type="video/ogg" /><!-- Firefox / Opera / Chrome10 --> -->
    <!-- fallback to Flash: -->
    <object width="512" height="512" type="application/x-shockwave-flash" data="__FLASH__.SWF">
      <!-- Firefox uses the `data` attribute above, IE/Safari uses the param below -->
      <param name="movie" value="__FLASH__.SWF" />
      <param name="flashvars" value="controlbar=over&amp;image=__POSTER__.JPG&amp;file=https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid0a_orbium_unicaudatus.mp4" />
      <!-- fallback image. note the title field below, put the title of the video there -->
      <img src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid0a_thumbnail.png" width="512" height="512" alt="thumbnail of _Orbium_ glider"
           title="No video playback capabilities, please download the video below" />
    </object>
  </video>
</div>
{:style="text-align:center;"}
  An <em>Orbium</em> glider in a Lenia CA ([Chan 2019](https://www.complex-systems.com/abstracts/v28_i03_a01/))
  <br>
  <strong>Download Video:</strong>
	<a href="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/vid0a_orbium_unicaudatus.mp4">"MP4"</a>
<!-- Open Format:	<a href="__VIDEO__.OGV">"Ogg"</a> -->

### We can decrease spatiotemporal discretization parameters `dt` and `1/kr` to more closely approximate the underlying mathematical ideal.
### Finer simulation comes at a computational cost.

{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/fractal_persistence/al24_slide_001) -- [Supporting resources](https://rivesunder.github.io/fractal_persistence) -- [Next slide](https://rivesunder.github.io/fractal_persistence/al24_slide_002b)
