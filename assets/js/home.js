$(function() {
  /* NOTE: hard-refresh the browser once you've updated this */
  $(".typed").typed({
    strings: [
      "stat sushant.daga<br/>" + 
      "><span class='caret'>$</span> skills: Deep Learning, Natural Language Processing, Computer Vision, Python, AWS, SQL ...<br/> ^100" +
      "><span class='caret'>$</span> job: Data Scientist at <a target='_blank' href='https://www.vmock.com/'>Vmock</a><br/> ^100" +
      "><span class='caret'>$</span> hobbies: Reading, Dancing, <a target='_blank' href='/writing'>Writing</a>, Badminton, TT<br/> ^300" +
      "><span class='caret'>$</span> miscellaneous: Research &amp; Data oriented, Experiment ready, Business minded <br/>"
    ],
    showCursor: true,
    cursorChar: '_',
    autoInsertCss: true,
    typeSpeed: 0.001,
    startDelay: 50,
    loop: false,
    showCursor: false,
    onStart: $('.message form').hide(),
    onStop: $('.message form').show(),
    onTypingResumed: $('.message form').hide(),
    onTypingPaused: $('.message form').show(),
    onComplete: $('.message form').show(),
    onStringTyped: function(pos, self) {$('.message form').show();},
  });
  $('.message form').hide()
});
