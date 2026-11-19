"""The words requests are made of, as data: synonyms of sound and job words, what adjectives do to the shared axes,
which words name a material, a voice, a domain or a direction. earshot.brief reads a request with these tables;
designs are found through the canonical words (the keys of CONCEPTS), so a design lists `uses=('success', ...)` once
and is found by "confirmation", "it worked" and "saved" too.

Every table is plain data. Add a word where it belongs; keep the canonical words few and concrete, and give a word to
one concept only (tests/test_brief.py checks that no synonym is listed twice). Plurals and verb forms are matched
through earshot.brief.stem, so 'chime' also finds 'chimes' and 'chiming'; list a form only when the stemmer cannot
reach it ('ran', 'shot').
"""

# canonical word -> other ways to say it (single words or short phrases, lower case)
CONCEPTS = {
    # ---- jobs in an interface
    'tap': ('press', 'pressed', 'pressing', 'button', 'button press', 'button tap', 'touch', 'push', 'push button',
            'cta', 'finger tap', 'tap feedback'),
    'click': ('clicky', 'mouse click', 'mouse', 'clack', 'click clack',
              'klick', 'klicken'),   # German, common in requests (no German lexicon: a few words)
    'hover': ('mouse over', 'mouseover', 'mousing over', 'mouse-over', 'rollover', 'roll over', 'focus', 'highlight',
              'cursor over', 'cursor move over', 'move over', 'moves over', 'hover over', 'pointer over',
              'mouse enter', 'hover state', 'hover effect', 'mouseenter'),
    'toggle': ('switch', 'toggle switch', 'checkbox', 'check box', 'radio button', 'flip', 'setting', 'on off',
               'flip switch', 'on/off', 'switch control', 'preference'),
    'switch_on': ('switch on', 'toggle on', 'turn on', 'turned on', 'enable', 'activate', 'engage', 'on switch',
                  'checked', 'turn something on', 'flip on', 'active', 'off to on', 'off to active', 'opt in',
                  'opt-in', 'power up the setting'),
    'switch_off': ('switch off', 'toggle off', 'turn off', 'turned off', 'disable', 'deactivate', 'disengage',
                   'unchecked', 'uncheck', 'deselect', 'off switch', 'turn something off', 'flip off', 'inactive',
                   'on to off', 'opt out', 'opt-out', 'mute'),
    'light_switch': ('light switch', 'lamp', 'lights', 'wall switch', 'flick the lights'),
    'open': ('expand', 'unfold', 'pop up', 'pops open', 'pop open', 'menu open', 'sheet open', 'show',
             'appear', 'enter', 'entrance', 'panel open', 'opening', 'slide up', 'bottom sheet', 'reveal panel',
             'show up', 'shows up', 'open up', 'unfurl', 'maximize', 'maximise', 'fan out', 'fans out'),
    'close': ('collapse', 'hide', 'dismiss', 'disappear', 'fold', 'minimize', 'minimise', 'shut', 'exit',
              'cancel', 'closing', 'put away', 'tuck away', 'tuck in', 'tuck-in', 'fold away', 'close up',
              'swipe away', 'go away', 'goes away', 'slide down'),
    'back': ('go back', 'previous', 'previous screen', 'previous page', 'return', 'retreat', 'back button',
             'navigate back', 'step back', 'going backwards', 'go backwards', 'backtrack', 'retrace', 'one step back',
             'back up a step'),
    'undo': ('redo', 'revert'),
    'next': ('forward', 'continue', 'proceed', 'advance', 'onward', 'onwards', 'next step', 'next page', 'next slide',
             'page change', 'tab', 'navigate', 'navigation', 'move ahead', 'moving ahead', 'go ahead', 'step forward',
             'setup wizard', 'switch between', 'switch tabs', 'tab change', 'change tabs'),
    'success': ('confirm', 'confirmation', 'confirmed', 'done', 'complete', 'completion', 'saved', 'save', 'ok',
                'okay', 'correct', 'correct answer', 'right answer', 'it worked', 'worked', 'approved', 'approve',
                'verified', 'verify', 'valid', 'task done', 'check mark', 'checkmark', 'tick mark', 'yes',
                'yep', 'yay', 'accepted', 'succeed', 'successful', 'successfully', 'go through', 'goes through',
                'went through', 'all good', 'payment success', 'finished', 'ticked off', 'tick off', 'checked off',
                'check off', 'a little yes', 'saying yes', 'well done', 'good job', 'great job', 'nice work',
                'all set', 'you are in', 'in the bag', 'nailed it', 'success state', 'thumbs up', 'got it right',
                'payment goes through', 'booking confirmed', 'order confirmed', 'sync complete', 'upload complete',
                'reassurance', 'sorted', 'erfolg', 'approval', 'acknowledgement', 'acknowledgment',
                'positive feedback'),
    'error': ('fail', 'failure', 'wrong', 'invalid', 'denied', 'deny', 'rejected', 'reject', 'incorrect',
              'wrong answer', 'forbidden', 'blocked', 'not allowed', 'mistake', 'oops', 'uh oh', 'uh-oh', 'nope',
              'declined', 'unauthorized', 'problem', 'failed login', 'login failed', 'saying no', 'says no',
              'say no', 'computer says no', 'try again', 'went wrong', 'goes wrong', 'something went wrong',
              'wrong password', 'access denied', 'error state', 'thumbs down', "can't do that",
              'connection lost', 'offline'),
    'warning': ('warn', 'caution', 'cautious', 'careful', 'attention', 'low battery', 'nearly full',
                'almost full', 'storage full', 'danger', 'hazard', 'watch out', 'hey careful', 'exceed', 'exceeds',
                'exceeded', 'limit', 'over the limit', 'speeding', 'running low', 'almost out', 'about to expire',
                'time out', 'timing out', 'times out', 'session timeout', 'about to time out', 'unsaved changes',
                'are you sure', 'beware', 'look out', 'might go wrong', 'could go wrong',
                'something is off', 'might be off', 'something might be wrong'),
    'notify': ('notification', 'notice', 'ping', 'reminder', 'remind', 'badge', 'toast notification',
               'toast message', 'inbox', 'mention', 'dm', 'push notification', 'heads up', 'heads-up',
               'new notification', 'psst', 'look here', 'hey look', 'tap on the shoulder', 'nudge', 'poke',
               'snackbar', 'banner', 'unread', 'toast',
               'you have mail', "you've got mail", 'new email', 'get attention',
               'grab attention', 'calendar event',
               'appointment', 'check-in', 'check in'),
    'receive': ('incoming', 'incoming message', 'message received', 'new message', 'new text', 'text you',
                'texts you', 'text back', 'reply', 'message arrive', 'arrive', 'arriving', 'download', 'receiving',
                'message comes in', 'comes in', 'chat message', 'got a message', 'got a text', 'new chat'),
    'send': ('sent', 'message sent', 'submit', 'outgoing', 'share', 'upload', 'fire off', 'send off',
             'fly away', 'flies away', 'whoosh away', 'sending', 'text sent', 'flown off', 'fly off', 'flew off',
             'zoom off', 'zoom away', 'email sent', 'off it goes', 'outbox', 'dispatch', 'goes out', 'go out',
             'went out'),
    'delete': ('remove', 'trash', 'bin', 'discard', 'clear', 'erase', 'archive', 'throw away', 'throwing away',
               'throw out', 'recycle', 'wastebasket', 'dispose', 'throw in the bin', 'throwing stuff in the bin',
               'get rid of', 'wipe', 'scrap', 'junk', 'garbage', 'rubbish', 'thrown away', 'thrown out'),
    'like': ('like button', 'favorite', 'favourite', 'favoriting', 'favouriting', 'favorited', 'favourited', 'fave',
             'heart icon', 'heart button', 'hearting', 'hearted', 'love', 'star', 'bookmark', 'upvote', 'reaction',
             'double tap', 'double-tap',
             'appreciate', 'kudos', 'wishlist', 'wish list'),
    'timer': ('alarm', 'time up', 'times up', 'time is up', 'timer done', 'timer alarm', 'kitchen timer', 'egg timer',
              'oven timer', 'oven', 'microwave', 'timer end', 'timer finished', 'alarm clock', 'wake up', 'wake-up',
              'wakeup', 'pomodoro', 'break over', 'session over', 'snooze'),
    'countdown': ('count down', 'counting down', 'race start', 'ready set go', 'three two one', 'traffic lights',
                  'traffic light', 'starting lights', 'start lights', 'before the start', 'beep beep beep'),
    'clock': ('clock tick', 'ticking clock', 'tick tock', 'ticktock', 'tick-tock', 'stopwatch', 'wall clock',
              'grandfather clock', 'running out of time', 'time running out', 'time is running out'),
    'tick': ('tic', 'detent', 'ratchet', 'notch', 'slider', 'slider tick', 'step by step', 'stepper', 'scrub',
             'scrubbing', 'count up', 'counts up', 'counting up', 'count-up', 'tick up', 'ticks up', 'ticking up',
             'snap into place', 'snaps into place', 'snapped into place', 'snapping into place', 'click into place',
             'clicks into place', 'clicked into place', 'clicking into place'),
    'type': ('typing', 'keystroke', 'key press', 'keypress', 'key click', 'keyboard', 'keyboard click', 'typewriter',
             'mechanical keyboard', 'text typing', 'typing text', 'type out', 'typed out', 'typewriter-style',
             'typewriter style', 'kinetic type', 'letter by letter', 'per letter', 'one letter at a time',
             'letter at a time', 'clacky', 'keys clacking', 'keyboard keys', 'tapping keys', 'typed', 'keycap',
             'text entry', 'input field'),
    'swipe': ('slide', 'scroll', 'drag', 'carousel', 'card slide', 'slide change', 'slide in', 'slide out',
              'brush aside', 'push aside', 'pushed aside', 'flick aside', 'swipe aside', 'swipe left', 'swipe right',
              'page through'),
    'refresh': ('reload', 'sync', 'pull to refresh', 'update', 'pull down to refresh', 'pulling down to refresh',
                'pulled down to refresh', 'pull-to-refresh'),
    'loading': ('load', 'progress', 'processing', 'thinking', 'waiting', 'spinner', 'busy', 'buffering'),
    'purchase': ('buy', 'bought', 'payment', 'pay', 'paid', 'checkout', 'order placed', 'cash register',
                 'transaction', 'sale', 'sold', 'ka-ching', 'kaching', 'cha-ching'),
    'add': ('add to cart', 'added', 'plus', 'new item', 'insert', 'create', 'created', 'basket', 'shopping cart',
            'shopping basket', 'shopping bag'),
    'unlock': ('login', 'log in', 'sign in', 'access granted', 'authenticated'),
    'lock': ('logout', 'log out', 'sign out', 'secure', 'lock up', 'locks up', 'locked up', 'locking up'),
    'record': ('recording', 'start recording', 'rec', 'record button', 'voice memo'),
    'shutter': ('camera shutter', 'camera', 'take a picture', 'take a photo', 'taking a picture', 'taking a photo',
                'snapshot', 'screenshot', 'screen capture', 'screen grab', 'slr', 'dslr', 'polaroid', 'photo snap',
                'capture', 'snap a photo', 'snap a picture', 'selfie'),
    'copy': ('copied', 'paste', 'pasted', 'duplicate', 'clipboard'),
    'startup': ('boot', 'boot up', 'app launch', 'launch app', 'welcome', 'app open', 'power on', 'start up',
                'greeting', 'hello', 'intro'),
    'shutdown': ('power off', 'goodbye', 'outro', 'quit', 'shut down'),
    'menu': ('menu move', 'menu navigation', 'menu item', 'cursor', 'cursor move', 'option', 'inventory', 'ui move',
             'scroll menu', 'menu blip'),
    'menu_select': ('menu select', 'select', 'selection', 'choose', 'pick option', 'press start', 'start button',
                    'select option', 'menu confirm', 'choice', 'lock in', 'locking in', 'pick a character',
                    'character select', 'make a choice'),
    'menu_back': ('menu back', 'back out', 'leave menu', 'exit menu', 'close menu', 'submenu'),
    # ---- jobs in a game
    'jump': ('hop', 'leap', 'boing', 'spring','double jump', 'trampoline', 'springboard', 'spring board',
             'bounce pad', 'jump pad', 'launch pad'),
    'land': ('landing', 'touch down', 'touchdown', 'drop down', 'hit the ground', 'hitting the ground',
             'hit the floor', 'hitting the floor', 'onto the floor', 'onto the ground', 'ground pound', 'fall down',
             'come down', 'came down', 'coming down', 'back on the ground', 'feet hit', 'lands on',
             # a card, a panel or a title that settles or comes to rest has landed, softly (JOB_FEEL): no falling
             # sweep (what winds down as it settles is a 'downlifter'); on a screen it is a soft thump (DOMAIN_SENSE)
             'settle', 'settle into', 'settles into', 'settled into', 'settling into', 'settle into place',
             'settles into place', 'come to rest', 'comes to rest', 'coming to rest', 'came to rest'),
    'footstep': ('foot step', 'step', 'walk', 'run', 'ran', 'stomp', 'stroll', 'pace', 'tread', 'stride',
                 'schritt', 'schritte'),
    'dash': ('dodge', 'dodge roll', 'dodge-roll', 'sprint', 'lunge', 'dart', 'speed burst', 'speed-burst',
             'air burst', 'burst of air'),
    'swing': ('sword swing', 'sword', 'greatsword', 'broadsword', 'longsword', 'claymore', 'slash', 'blade', 'katana', 'axe swing', 'club swing', 'sword slash'),
    'hit': ('punch', 'jab', 'uppercut', 'strike', 'smack', 'slap', 'attack', 'melee', 'bash', 'whack', 'sock',
            'socked', 'thwack', 'fist', 'fighting', 'brawl', 'beat up', 'karate kick', 'roundhouse kick', 'flying kick',
            'land a punch', 'land a hit', 'land a jab',
            'land a blow', 'landing a punch', 'landing a hit', 'landing a jab', 'landing a blow',
            'boxer', 'boxing', 'punching bag',
            # one enemy of many beaten is the blow that beats it, played all through a level: no fanfare (a boss
            # defeated is a win), no tune of losing (the player's own death is)
            'enemy defeated', 'enemy is defeated', 'enemy killed', 'enemy is killed', 'enemy dies', 'enemy death',
            'enemy destroyed', 'enemy is destroyed', 'enemy down', 'defeat an enemy', 'defeating an enemy',
            'kill an enemy', 'killing an enemy', 'enemy kill'),
    'clang': ('clank', 'clonk', 'metal hit', 'metal impact', 'metallic hit', 'sword hit', 'clash', 'sword clash',
              'sword on armour', 'sword on armor', 'sword on shield', 'blade on metal'),
    'hurt': ('damage', 'stung', 'bee sting', 'bitten', 'scratched', 'burned', 'take damage', 'taking damage',
             'lose health', 'loses health', 'losing health', 'ouch', 'injured', 'injury', 'player hit', 'pain',
             'hp loss', 'health loss', 'get hit', 'gets hit', 'got hit'),
    'lose': ('lost', 'game over', 'death', 'die', 'died', 'dying', 'dead', 'defeat', 'last life', 'lose a life',
             'fail state', 'failed level', 'level failed', 'you died', 'fails a level', 'fail a level',
             'run out of lives', 'out of lives', 'no lives left', 'try again screen', 'wipe out', 'wiped out',
             'killed', 'eliminated', 'sad trombone', 'womp womp', 'out of moves', 'run out of moves', 'no moves left',
             'out of time', 'trombone', 'wah wah', 'wah-wah'),
    'win': ('won', 'victory', 'victorious', 'level complete', 'level cleared', 'stage clear', 'stage complete',
            'beat the level', 'beat a stage', 'clear a stage', 'clear the level', 'triumph', 'flourish', 'triumphant',
            'fanfare', 'celebration', 'celebrate', 'winner', 'winning tune', 'ta-da', 'tada', 'ta da',
            'finish line', 'first place', 'win the race', 'round won', 'champion', 'boss defeated',
            'defeat the boss', 'beat the boss', 'boss down', 'boss beaten',
            'celebratory', 'congratulations', 'congrats', 'jackpot', 'quest complete', 'mission complete'),
    'level_up': ('levelup', 'level-up', 'leveled up', 'levelled up', 'rank up', 'reach level', 'new level', 'xp',
                 'rank', 'gain a rank', 'gain a level', 'gains a level', 'next level', 'skill up', 'got better',
                 'get better', 'getting better', 'got stronger', 'get stronger', 'getting stronger', 'promotion',
                 'promoted', 'evolve', 'evolution'),
    'one_up': ('extra life', '1up', '1-up', 'one-up', 'bonus life'),
    'achievement': ('trophy', 'reward', 'badge earned', 'achievement unlocked', 'trophy unlocked', 'badge unlocked',
                    'unlocked an achievement', 'unlock an achievement'),
    'coin': ('coin pickup', 'collect coin', 'gold', 'gold coin', 'money', 'token', 'ring pickup', 'cash pickup',
             'score', 'cash', 'rings', 'currency', 'bucks', 'coins', 'münze', 'münzen'),
    'gem': ('jewel', 'diamond', 'crystal pickup', 'treasure', 'rare item', 'gemstone', 'ruby', 'emerald', 'sapphire',
            'crystal shard', 'precious stone', 'jewels'),
    'pickup': ('pick up', 'collect', 'collectible', 'grab', 'item', 'loot', 'get item', 'acquire', 'got it',
               'find', 'found', 'snag', 'scoop up', 'obtain', 'ammo', 'powerup crate'),
    'power_up': ('power up', 'getting powerful', 'get powerful', 'powering up', 'rage', 'rage mode', 'berserk',
                 'frenzy', 'overdrive', 'powerup', 'power-up', 'powered up', 'boost', 'speed boost', 'upgrade', 'buff',
                 'charge up', 'charging up', 'charging', 'charged', 'grow', 'mushroom', 'super strong', 'invincible',
                 'invincibility', 'star power'),
    'power_down': ('power down', 'powerdown', 'shrink', 'debuff', 'lose power', 'loses power', 'drain', 'wear off',
                   'wears off', 'wearing off', 'run out', 'runs out', 'expire', 'fizzle', 'fizzle out',
                   'fizzles out', 'lose the buff', 'losing the buff', 'weaken',
                   'weakened', 'running on empty', 'out of power'),
    'heal': ('health', 'health restore', 'restore', 'hp', 'regenerate', 'regen', 'potion', 'revive', 'cure',
             'recover', 'recovery', 'first aid', 'medkit', 'life back', 'health back', 'hp back', 'refill health',
             'patch up', 'mend', 'healed up', 'feel better'),
    'shield': ('block', 'parry', 'guard', 'deflect', 'armor', 'armour', 'barrier', 'force field', 'buckler',
               'bounce off', 'bouncing off', 'bounces off', 'glance off', 'ricochet off', 'stop a hit',
               'stop an arrow', 'stopping an arrow', 'blocked hit'),
    'teleport': ('teleporter', 'teleporting', 'warp', 'portal', 'blink',
                 'beam up', 'beam me up', 'beam-me-up', 'respawn', 'spawn', 'vanish',
                 'reappear', 'materialize', 'materialise', 'dematerialize', 'poof', 'warpy', 'disappear and reappear',
                 'gone in a flash', 'fast travel', 'phase out', 'blink away'),
    'magic': ('magical', 'mystical', 'spell', 'cast', 'spell cast', 'enchant', 'enchanted', 'enchantment', 'witch',
              'sorcery', 'sorcerer', 'mana', 'arcane', 'summon', 'fireball', 'wand', 'incantation', 'wizard',
              'waving a wand',
              'wave a wand'),
    'secret': ('hidden', 'hidden item', 'discover', 'discovery', 'mystery', 'secret area', 'easter egg', 'mysterious',
               'puzzle solved'),
    'checkpoint': ('save point', 'save spot', 'save station', 'rest point', 'safe spot', 'progress saved',
                   'progress is safe', 'progress safe', 'autosave', 'auto save', 'saving progress', 'flag',
                   'milestone'),
    'explosion': ('explode', 'blast', 'bomb', 'detonation', 'detonate', 'kaboom', 'grenade', 'blow up',
                  'blowing up', 'blown up', 'blew up', 'firecracker', 'burst', 'noise burst', 'firework',
                  'fireworks', 'dynamite', 'tnt', 'nuke', 'detonator', 'landmine'),
    'shoot': ('shot', 'gun', 'gunshot', 'gunfire', 'fire weapon', 'firing', 'pistol', 'handgun', 'revolver', 'rifle',
              'shotgun', 'cannon', 'bullet', 'projectile', 'crossbow', 'musket', 'sniper',
              'fire a weapon', 'pull the trigger'),
    # an arrow is heard where it lands (the recorded arrow/hit), not as a gunshot
    'arrow': ('arrows', 'archery', 'archer', 'arrow hit'),
    'laser': ('blaster', 'zap', 'pew', 'pew pew', 'pew-pew', 'ray gun', 'raygun', 'beam', 'phaser', 'plasma'),
    'door': ('gate', 'hatch', 'door open', 'door close', 'creak', 'creaky', 'door slam', 'latch',
             'tür'),
    'chest': ('treasure chest', 'chest open', 'crate open', 'box open', 'open a box', 'unbox', 'unboxing',
              'loot box', 'lootbox', 'loot crate', 'lid', 'goodies', 'box of goodies', 'treasure box'),
    'break': ('shatter', 'smash', 'destroy', 'crumble', 'glass break', 'wood break', 'splinter', 'broke',
              'into pieces', 'to pieces', 'into bits', 'to bits', 'into shards', 'cracking into', 'snap in half',
              'snapping in half', 'fall apart', 'falls apart', 'smithereens', 'wreck', 'demolish'),
    'splash': ('splashy', 'water splash', 'dive', 'plunge', 'sploosh', 'splosh', 'cannonball', 'swim',
               'fall into water', 'falling into water', 'drop into water', 'jump into water', 'into the water',
               'into a pool', 'into the lake', 'puddle', 'belly flop', 'wade', 'wading', 'paddle'),
    'engine': ('motor', 'car', 'vehicle', 'thruster', 'rocket'),
    'text': ('dialogue', 'dialog text', 'text box', 'text reveal', 'text appear', 'subtitle', 'letter', 'speech',
             'talk', 'npc talk', 'chat bubble'),
    # ---- gestures of film and motion design
    'whoosh': ('swoosh', 'woosh', 'air movement', 'transition', 'rush of air', 'air rush', 'wind rush', 'wind-rush',
               'fly through', 'flying through', 'fly across', 'flies across', 'fly in', 'whip pan', 'whooshy',
               'rushing air', 'air sweep', 'rushing', 'blowing past', 'blow past', 'rushing past', 'zoomy',
               'fly off screen', 'slides off', 'swooshy', 'wusch', 'übergang',
               # the camera of a film moving fast (a zoom), something leaving the frame
               'zoom', 'zoom in', 'camera zoom', 'flies off',
               # the camera moving through the shot
               'camera push in', 'camera pushes in', 'camera pushing in', 'camera pushes into', 'camera moves into',
               'camera pull out', 'camera pulls out',
               'camera pulling out', 'camera pan', 'camera pans', 'camera panning', 'camera move', 'camera moves',
               'camera sweeps', 'push in on', 'pushes in on', 'pull back to reveal', 'pulls back to reveal'),
    # something carried past the listener (a whoosh too: BROADER)
    'flyby': ('fly by', 'fly-by', 'flyover', 'fly over', 'fly overhead', 'flying overhead', 'flies overhead',
              'passing overhead', 'flies past', 'flying past', 'fly past', 'whoosh by',
              'pass by', 'passing by', 'pass', 'zoom by', 'zoom past', 'zooming past', 'streak past', 'drive by',
              'drive-by'),
    'swish': ('swoop', 'flick', 'whip', 'whippy', 'quick whoosh', 'fast pass', 'zing', 'swishy', 'fwip', 'fwoop',
              'vwip', 'shwip',
              # a line or stroke drawn on by itself in a film
              'line draws on', 'line draw on', 'line drawing on', 'stroke draws on', 'stroke draw on'),
    'riser': ('uplifter', 'build', 'build up', 'build-up', 'buildup', 'tension', 'tension build', 'anticipation',
              'crescendo', 'suspense', 'suspenseful', 'ramp up', 'swell', 'keeps climbing', 'keep climbing',
              'getting tenser', 'getting louder', 'about to happen', 'lead in', 'lead-in', 'ramp', 'rising tension'),
    'downlifter': ('down lifter', 'downer', 'wind down', 'winding down', 'decelerate', 'sink',
                   # what winds down when it settles: the energy, the dust, a crowd (a thing that settles has
                   # landed: 'land')
                   'settle down', 'settles down', 'settled down', 'settling down', 'energy settles',
                   'energy settling', 'dust settles', 'dust settling',
                   'fall away', 'fade away', 'fades away', 'falling sweep', 'down sweep', 'sweep down', 'pitch drop',
                   'pitch dropping', 'pitch-dropping', 'tension release', 'release of tension', 'calm down',
                   'calming down', 'drop away', 'dropping away', 'drops away', 'die down', 'dies down',
                   'come down from',
                   'power down sweep', 'tape stop', 'tape slowdown', 'tape slow down', 'record stop'),
    'impact': ('boom hit', 'cinematic hit', 'trailer hit', 'slam', 'slam hit', 'big hit', 'bang', 'crash',
               'collision', 'hit on a cut', 'title slam', 'boom', 'wham', 'whomp'),
    'thud': ('thump', 'thunk', 'thwump', 'dull hit'),
    'sub_drop': ('sub drop', 'sub-drop', 'bass drop', 'sub boom', 'low boom', 'sub hit', '808 drop', 'low-end drop',
                 'bass hit', 'bass impact',
                 'low end drop', 'sub bass drop', 'deep drop', 'low drop', 'bass thump', 'feel more than hear',
                 'chest thump', 'sub thump'),
    'braam': ('bwaaam', 'bwaam', 'braaam', 'horn blast', 'brass blast', 'trailer horn', 'inception horn', 'horn hit',
              'brass hit'),
    'logo': ('logo reveal', 'logo sting', 'brand', 'branding', 'sonic logo', 'audio logo', 'sound logo', 'mnemonic',
             'ident', 'signature', 'sonic identity', 'brand sound', 'company logo', 'audio branding'),
    'sting': ('stinger', 'music sting', 'musical sting', 'stab sting'),
    'glitch': ('glitchy', 'stutter', 'digital noise', 'error glitch', 'distortion', 'datamosh', 'static burst',
               'malfunction', 'freaking out', 'freak out', 'bitcrush', 'corrupted', 'data corruption', 'flicker',
              'flickering', 'light flicker', 'neon flicker'),
    'shimmer': ('sparkle', 'twinkle', 'twinkly', 'glitter', 'glittery', 'shine', 'glint', 'gleam', 'fairy dust',
                'fairy-dust', 'pixie dust', 'magic dust', 'fairy'),
    'pop': ('bubble pop', 'plop', 'pop in', 'pop-in', 'popin', 'bloop', 'boop', 'soap bubble', 'blop',
            'bop', 'pip up', 'boing in', 'scale in', 'scales in', 'scaled in', 'scaling in', 'bounce in',
            'bounces in', 'bounced in', 'bouncing in'),
    'reverse': ('reverse cymbal', 'reverse swell', 'reverse whoosh', 'suck', 'suck back', 'rewind', 'backwards',
                'inhale'),
    'pad': ('synth pad', 'atmosphere pad', 'sustained chord', 'held chord', 'chord bed', 'background chord',
            'ambient pad', 'warm pad'),
    'drone': ('tension bed', 'dark ambience', 'low drone'),
    'rumble': ('earthquake', 'quake', 'tremor', 'ground shake'),
    # ---- sources
    'bell': ('ding', 'dong', 'ding dong', 'chime', 'ting', 'tinkle', 'bell strike', 'bell hit', 'struck bell',
             'ring', 'ringing bell'),
    'service_bell': ('service', 'service bell', 'desk bell', 'hotel bell', 'reception bell', 'counter bell',
                     'reception', 'concierge', 'order up'),
    'knock': ('tock', 'thock', 'wood knock', 'rap', 'tok', 'door knock', 'knock knock', 'knock on the door',
              'klopfen', 'anklopfen'),
    'beep': ('bleep', 'bleepy', 'beepy', 'pip', 'blip'),
    'buzz': ('buzzer', 'buzzy', 'raspberry', 'blat', 'grumpy honk'),
    'hum': ('humming', 'whir', 'whirr', 'drone hum'),
    'crackle': ('crackling', 'crackly'),
    'pluck': ('plucky', 'string pluck', 'strum', 'plink', 'plinky', 'twang'),
    'piano': ('grand piano', 'upright piano', 'felt piano', 'piano key', 'piano note', 'piano tone'),
    'bass': ('bass note', 'sub', 'sub bass', 'low end', 'bassline', '808 bass', 'bass string'),
    # ---- drums and percussion (a family each; 'crash' in a request is a collision unless drums are around it)
    'drum': ('percussion', 'drum hit', 'drums', 'drum kit', 'drum sound', 'percussive', 'acoustic drum'),
    'kick_drum': ('kick drum', 'kick', 'bass drum', 'kick 808', '808 kick', '808', 'four on the floor',
                  'kick on the downbeat'),
    'snare': ('snare drum', 'snares', 'backbeat', 'snare hit', 'snare crack'),
    'rimshot': ('rim shot', 'rim'),
    'cross_stick': ('cross-stick', 'cross stick', 'side stick', 'sidestick', 'rim click', 'rim knock'),
    'tom': ('tom tom', 'tom-tom', 'floor tom', 'rack tom', 'drum fill', 'tom fill'),
    'hihat': ('hi-hat', 'hi hat', 'hat', 'closed hat', 'open hat', 'hh', 'hats'),
    'cymbal': ('cymbals', 'crash cymbal', 'ride cymbal', 'splash cymbal', 'china cymbal', 'cymbal crash',
               'ride bell', 'cymbal hit'),
    'clap': ('handclap', 'hand clap', 'clapping', 'claps'),
    # many people clapping is an audience, not a row of claps (a recording: applause/)
    'applause': ('applauding', 'applaud', 'applauds', 'ovation', 'standing ovation', 'round of applause',
                 'clapping audience', 'clapping crowd',
                 'audience applause', 'people applauding'),
    'finger_snap': ('finger snap', 'snap fingers', 'snapping fingers', 'finger click', 'fingersnap',
                    'snap your fingers', 'snapping your fingers'),
    'shaker': ('egg shaker', 'maraca', 'maracas'),
    'tambourine': ('tamb', 'timbrel'),
    'conga': ('congas', 'tumba'),
    'bongo': ('bongos',),
    'darbuka': ('darbouka', 'doumbek', 'goblet drum'),
    'frame_drum': ('frame drum', 'bodhran'),
    'cajon': ('box drum',),
    'clave': ('claves',),
    'cabasa': ('afuche',),
    'slapstick': ('slap stick',),
    'drum_roll': ('drum roll', 'drumroll', 'snare roll', 'press roll'),
    'count_in': ('count in', 'count-in', 'count off', 'drumsticks', 'stick click', 'sticks clicking'),
    'note': ('single note', 'one note', 'tone'),
    'paper': ('page', 'sheet', 'sheet of paper', 'card', 'envelope', 'rustle', 'paper rustle', 'flip through',
              'flipping through', 'notebook', 'book', 'page turn', 'page flip', 'flip pages', 'flipping pages',
              'turn the page', 'turning pages', 'leaf through'),
    'water': ('drip', 'droplet', 'water drop', 'water droplet', 'drop of water', 'liquid', 'leaky tap',
              'leaky faucet', 'tap drip'),
    # a tap that water runs from (a recording): "a tap running" is no press and no walk
    'faucet': ('water tap', 'tap water', 'tap running', 'running tap', 'kitchen tap', 'bathroom tap',
               'water from a tap', 'running water from a tap', 'water running from a tap', 'turn on the tap',
               'turning on the tap'),
    'coin_object': ('coin drop', 'coin dropping', 'dropping a coin', 'drop a coin', 'coin spin', 'coin on table',
                    'coin flip', 'coin toss', 'penny', 'loose change', 'coin rattle', 'small change', 'spare change'),
    'glass_object': ('clink', 'cheers', 'toast glass', 'glass tap', 'glasses clink', 'wine glass', 'glass clink',
                     'champagne flutes', 'champagne flute', 'flutes', 'tapping together', 'clinking together',
                     'raise a glass', 'propose a toast', 'wedding toast', 'toasting'),
    'cork': ('cork pop', 'champagne', 'champagne pop', 'bottle pop', 'pop the cork', 'uncork', 'pop a bottle',
             'bubbly bottle', 'popping the bubbly'),
    'keyring': ('key ring', 'keychain', 'jingle keys', 'keys jingle', 'jingling keys', 'bunch of keys', 'car keys',
                'house keys'),
    'scissors': ('snip', 'cutting paper'),
    'heartbeat': ('heart', 'heart beat', 'heart pounding', 'heartbeat pulse', 'beating heart'),
    # a phone or a game pad shaking in the hand (a phone buzzing on a table): the recording, not a buzzer's tone
    'vibrate': ('vibration', 'vibrating', 'haptic', 'haptics', 'haptic feedback'),
    # ---- places and weather
    'storm': ('stormy', 'storm outside', 'distant storm'),
    'rain': ('rainfall', 'drizzle', 'downpour', 'rainy', 'pitter-patter', 'pitter patter', 'rainstorm',
             'shower', 'raindrop', 'rain drop', 'rain shower'),
    'wind': ('windy', 'breeze', 'breezy', 'gust', 'gusty', 'gale', 'howling', 'blowing', 'blustery', 'whistly',
             'windstorm', 'draught', 'howl', 'clifftop', 'cliff top', 'wind blowing'),
    'fire': ('campfire', 'fireplace', 'flame', 'burning', 'bonfire', 'torch', 'logs', 'burning logs', 'hearth',
             'fire crackle'),
    'stream': ('river', 'creek', 'brook', 'waterfall', 'running water', 'flowing water', 'babbling', 'trickle',
               'rapids'),
    'waves': ('ocean', 'sea', 'beach', 'surf', 'shore', 'tide', 'seaside', 'coast', 'coastal', 'shoreline',
              'rocky shore', 'rocks', 'shingle', 'pebble beach'),
    'cafe': ('café', 'coffee shop', 'restaurant', 'bistro', 'diner', 'bar', 'pub', 'crowd murmur', 'chatter',
             'walla'),
    'forest': ('birds', 'birdsong', 'birds singing', 'woods', 'woodland', 'jungle', 'nature', 'meadow', 'garden',
               'birds chirping', 'songbirds', 'tweeting', 'park'),
    'night': ('crickets', 'cricket', 'insects', 'evening', 'summer night', 'cicadas', 'frog chorus', 'bugs',
              'after dark', 'nighttime', 'night time', 'midnight', 'dusk', 'night sounds'),
    'thunder': ('lightning', 'thunderstorm', 'thunderclap', 'thunder clap', 'rolling thunder', 'rolling rumble'),
    'room_tone': ('room tone', 'silence', 'quiet room', 'office', 'indoor ambience', 'air conditioning', 'indoor',
                  'indoors', 'empty room', 'hvac', 'ventilation'),
    'city': ('traffic', 'street', 'urban', 'downtown', 'town', 'busy town', 'city street', 'metropolis'),
    'space_hum': ('spaceship', 'space station', 'engine room', 'sci-fi ambience', 'machine room', 'server room',
                  'reactor', 'starship', 'space ship', 'ship interior', 'machine hum', 'machine humming',
                  'big machine', 'machine', 'mechanical whir', 'mechanical hum'),
    'underwater': ('under water', 'submerged', 'deep sea', 'under the water', 'below water', 'bottom of the pool',
                   'bottom of the sea', 'bottom of the ocean', 'bottom of the lake', 'beneath the surface',
                   'underwater bubbles', 'bubbles', 'scuba', 'snorkel', 'diving deep', 'head under water'),
    'ambience': ('ambient', 'background', 'atmosphere', 'soundscape', 'environment', 'backdrop', 'atmos',
                 'background bed', 'bed'),
    # ---- music under something: a bed with chords, a beat or a tune, not one note and not an ambience
    'music': ('background music', 'music bed', 'bed music', 'soundtrack', 'underscore', 'film score', 'music score',
              'backing track', 'background track', 'music loop', 'music track', 'lo-fi beat', 'lofi beat',
              'lo-fi music', 'lofi music', 'lo-fi hip hop', 'beat', 'beats', 'jingle', 'menu music', 'game music',
              'level music', 'elevator music', 'hold music', 'ambient music', 'chill music', 'instrumental', 'bgm',
              'drum loop', 'beat loop',
              'music for a video', 'video music', 'theme music', 'title music', 'loading music'),
    # ---- what a request calls any short sound: names the kind, not the job
    'cue': ('alert', 'alert tone', 'audio cue', 'sound cue', 'signal', 'earcon', 'ui sound', 'feedback sound',
            'tune', 'melody'),
    # ---- what animals, a crowd and a few household things do (recordings: earshot.sounds.learned, FAMILY_SENSE)
    'bark': ('barking', 'barks', 'woof', 'woofs', 'yap', 'yapping', 'yip'),
    'meow': ('meowing', 'meows', 'miaow', 'mew', 'mewing'),
    'moo': ('mooing', 'moos', 'lowing'),
    # one bird is a bird's call ("a bird chirping"); birds, plural, are the forest's bed
    'chirp': ('chirping', 'chirps', 'tweet', 'twitter', 'twittering', 'bird', 'bird call', 'birdcall', 'robin',
              'sparrow', 'blackbird', 'songbird', 'bird chirping', 'bird singing', 'bird sings'),
    'honk': ('honking', 'honks', 'horn', 'car horn', 'car honk', 'horn honk', 'toot', 'tooting', 'hooter'),
    'cry': ('crying', 'cries', 'wail', 'wailing', 'weep', 'weeping', 'whimper', 'whimpering', 'tears'),
    # a phone named as the thing that rings (a plain "phone" is often only the device a job happens on)
    'telephone': ('phone call', 'incoming call', 'phone ringing', 'phone rings', 'ringing phone', 'landline',
                  'rotary phone', 'desk phone', 'old phone', 'telephone ringing'),
    'hoot': ('hooting', 'hoots'),
    'caw': ('cawing', 'caws'),
    'cheer': ('cheering', 'cheers and applause', 'whooping'),
    'sizzle': ('sizzling', 'sizzles', 'frying'),
    'pour': ('pouring', 'poured', 'pours'),
    'flush': ('flushing', 'flushes'),
    # one frog (a recording); a chorus of them is the night's bed
    'croak': ('ribbit', 'toad', 'frog croaking', 'croaking frog', 'frog call'),
    # a gun loaded again (a recording); a page loaded again is a refresh (DOMAIN_SENSE, WORD_WITH)
    'gun_reload': ('gun reload', 'reload a gun', 'reloading a gun', 'weapon reload', 'reload a weapon', 'cock a gun',
                   'rack the slide'),
    # ---- what is done in a kitchen (recordings)
    'chop': ('dicing', 'chopping board', 'cutting board'),
    'boil': ('simmer', 'boiling water', 'water boiling', 'rolling boil'),
    'stir': ('teaspoon',),
    'ignite': ('igniter', 'gas stove', 'gas burner', 'gas hob', 'gas cooker', 'burner', 'stove lighting',
               'light the stove', 'lighting the stove'),
}

# sounds the catalogue has nothing of: voices that say or sing words, a shout, sobbing (a scream, a sneeze, a burp
# and the farm animals are recordings: earshot.sounds.learned), instruments no design plays, and eating and kitchen
# things no recording plays (chewing, a kettle, a fridge). A request that names one is answered with the nearest
# design and says so (make.doubt), not with a confident wrong sound. Exact words, as written, and only where the
# reader understood them as nothing else (brief.not_made: "saying yes" is a success). A word of this list inside a
# whole phrase a design is named for is that design ("an elephant trumpeting" is no trumpet)
NOT_MADE = frozenset((
    'yell', 'yelling', 'shout', 'shouting', 'sob', 'sobbing',
    'singing', 'sings', 'singer', 'vocal', 'vocals', 'choir', 'spoken', 'saying', 'narrator', 'narration',
    'voiceover', 'voice-over',
    'violin', 'violins', 'cello', 'viola', 'saxophone', 'sax', 'flute', 'clarinet', 'oboe', 'bagpipes',
    'trumpet', 'chew', 'chewing', 'munch', 'munching', 'airlock'))
NOT_MADE_PHRASES = ('air horn', 'electric guitar', 'guitar solo', 'drum solo', 'ringtone', 'kitchen timer',
                    'egg timer', 'fire alarm', 'smoke alarm', 'car alarm', 'burglar alarm', 'toy gun', 'water gun',
                    'nerf gun', 'measuring tape',
                    'tape measure', 'pan set', 'machine gun',
                    'pan placed', 'pan put down', 'saucepan set', 'saucepan placed', 'saucepan put down')
# a voice the catalogue has of one kind of person only (the recording's): said of another person it is not made ("a
# woman crying": the crying is a baby's; brief.not_made)
ONE_VOICE = {'cry': frozenset(('baby', 'babies', 'infant', 'infants', 'newborn', 'toddler'))}
PERSON_WORDS = frozenset(('woman', 'women', 'man', 'men', 'girl', 'girls', 'boy', 'boys', 'adult', 'adults', 'lady',
                          'guy', 'person', 'teenager', 'teen', 'mother', 'father', 'mom', 'dad', 'wife', 'husband',
                          'actor', 'actress', 'soldier'))
# a word whose job changes when the request also says one of these words: {word: (words, concept)} ("a toast" in an
# app is a notification; "a toast at a wedding", "cups going clink for a toast" are glasses; brief.read)
WORD_WITH = {'toast': (frozenset(('glass', 'glasses', 'cup', 'cups', 'clink', 'clinking', 'cheers', 'champagne',
                                  'wine', 'wedding', 'drink', 'drinks', 'bride', 'groom', 'party', 'raise',
                                  'raising')), 'glass_object'),
             # "shoot an arrow": the arrow, not a gun
             'shoot': (frozenset(('arrow', 'arrows')), 'arrow'),
             # "a heart" is a heartbeat; a heart tapped on a post or a song is a like
             'heart': (frozenset(('tap', 'taps', 'tapped', 'tapping', 'icon', 'button', 'click', 'clicked', 'post',
                                  'song', 'photo', 'app', 'feed')), 'like'),
             # "a clink" is two glasses; the clink of a key or of keys is the keys'
             'clink': (frozenset(('key', 'keys', 'keyring', 'keychain')), 'keyring'),
             # a gun, not a page, is loaded again
             'reload': (frozenset(('gun', 'guns', 'weapon', 'rifle', 'pistol', 'shotgun', 'ammo', 'magazine',
                                   'shooter', 'fps')), 'gun_reload'),
             # the camera of a film that moves is no photo taken
             'camera': (frozenset(('pushes', 'pushing', 'pans', 'panning', 'pulls', 'pulling', 'moves', 'moving',
                                   'zooms', 'zooming', 'tracks', 'tracking', 'dollies', 'dolly', 'sweeps', 'orbits',
                                   'tilts', 'glides', 'drifts', 'flies')), 'whoosh'),
             # what bubbles in a pot boils; bubbles alone are under water
             'bubbl': (frozenset(('pot', 'soup', 'stew', 'sauce', 'saucepan', 'stove', 'hob', 'kettle')), 'boil')}
# a word that is no job at all when the request also says one of these words: toast bitten, buttered or out of a
# toaster is bread, not a notification (brief.read: the word is then read as a plain word)
WORD_WITHOUT = {'toast': frozenset(('bite', 'bites', 'biting', 'bitten', 'crunch', 'crunchy', 'crunching', 'bread',
                                    'butter', 'buttered', 'toaster', 'breakfast', 'eat', 'eating', 'jam', 'crispy',
                                    'burnt', 'slice')),
                # water or a machine that runs takes no steps
                'run': frozenset(('faucet', 'water', 'engine', 'motor', 'machine', 'fridge', 'shower', 'hose',
                                  'dishwasher', 'fan')),
                # a dart thrown at a board is no quick dodge; ping pong is a game, not a notification
                'dart': frozenset(('board', 'dartboard', 'bullseye', 'thrown', 'throw', 'throwing', 'hitting', 'hits')),
                'ping': frozenset(('pong',))}
# a job done with the same thing as another one's family: a recorded lock answers an unlocking too (brief.learned_ok)
SAME_THING = {'unlock': 'lock'}
# what many people doing a thing at once is ("people clapping": an audience), unless they do it as one ("clapping
# together": one clap of many hands); brief.read
CROWD_OF = {'clap': 'applause'}
GROUP_WORDS = frozenset(('people', 'crowd', 'audience', 'everyone', 'everybody', 'room', 'spectators', 'fans',
                         'guests', 'class', 'hall', 'stadium', 'listeners', 'viewers'))
UNISON_WORDS = frozenset(('together', 'unison', 'once', 'single', 'one', 'sync'))
# when a sound lands, not what it is: "a hit on the beat" asks for no music (brief.read skips these)
TIMING_PHRASES = ('on the beat', 'on beat', 'to the beat', 'with the beat', 'on each beat', 'on every beat',
                  'in time with the beat', 'in sync with the beat', 'in time with the music')
# concepts that say what kind of thing is wanted without saying which: they do not end the head of a request
GENERIC = frozenset(('ambience', 'note', 'beep', 'bell', 'open', 'achievement', 'cue', 'thud', 'sting'))
# words of a short sound: an instrument named just before one is played ("a kalimba ping", "a marimba hit")
SOUND_NOUNS = frozenset(('ping', 'ding', 'chime', 'plink', 'tone', 'hit', 'note', 'pluck', 'tap', 'tick', 'knock',
                         'bonk', 'plonk', 'strike', 'pling', 'tink', 'blip', 'boop', 'beep', 'sound', 'cue'))
# a background asked for in a dark mood is a drone ("an ominous low bed")
DARK_MOODS = frozenset(('ominous', 'dark', 'creepy', 'eerie', 'sinister', 'unsettling', 'scary', 'spooky', 'tense',
                        'menacing', 'foreboding'))
# adjectives of a concept colour a sound ("a windy zip", "a buzzy thud"): a part of it, never its head
COLOUR_WORDS = frozenset(('windy', 'breezy', 'gusty', 'blustery', 'rainy', 'stormy', 'whistly', 'buzzy', 'glitchy',
                          'splashy', 'crackly', 'clicky', 'bleepy', 'beepy', 'whooshy', 'swooshy', 'swishy', 'twinkly'))
# "old Nintendo games", "an ancient home computer": an old-sounding word near a device word means chip sound
OLD_WORDS = frozenset(('old', 'ancient', 'vintage', 'classic', 'early', '80s', '90s', 'eighties', 'nineties',
                       'oldschool', 'antique', 'nostalgic'))
OLD_DEVICES = frozenset(('game', 'games', 'computer', 'computers', 'console', 'consoles', 'calculator', 'handheld',
                         'arcade', 'cartridge', 'pc', 'micro', 'nintendo', 'atari', 'videogame', 'videogames'))
# separable verbs: verb ... particle (up to three words apart: "picking something up", "putting a window away")
PHRASAL = {'put': {'away': 'close', 'down': 'land'}, 'tuck': {'away': 'close', 'in': 'close'},
           'pick': {'up': 'pickup'}, 'turn': {'on': 'switch_on', 'off': 'switch_off'},
           'switch': {'on': 'switch_on', 'off': 'switch_off'}, 'flip': {'on': 'switch_on', 'off': 'switch_off'},
           'flick': {'on': 'switch_on', 'off': 'switch_off'}, 'throw': {'away': 'delete', 'out': 'delete'},
           'send': {'off': 'send', 'out': 'send'}, 'fire': {'off': 'send'}, 'wake': {'up': 'timer'},
           'power': {'up': 'power_up', 'down': 'power_down'}, 'shut': {'down': 'shutdown'},
           'blow': {'up': 'explosion'}, 'level': {'up': 'level_up'}, 'calm': {'down': 'downlifter'},
           'boot': {'up': 'startup'}, 'log': {'in': 'unlock', 'out': 'lock'}, 'sign': {'in': 'unlock', 'out': 'lock'},
           'mouse': {'over': 'hover', 'on': 'hover', 'onto': 'hover'},
           'pointer': {'over': 'hover', 'on': 'hover', 'onto': 'hover'},
           'cursor': {'over': 'hover', 'on': 'hover', 'onto': 'hover'},
           'zoom': {'past': 'flyby', 'by': 'flyby', 'off': 'send'}, 'fly': {'past': 'flyby', 'by': 'flyby'},
           'brush': {'aside': 'swipe', 'away': 'swipe'}, 'push': {'aside': 'swipe', 'away': 'swipe'},
           'shrug': {'off': 'shield'}, 'knock': {'over': 'hit'}, 'wipe': {'out': 'lose', 'away': 'delete'},
           'set': {'down': 'land'}}
# outcome words: named after the thing they happen to ("upload finished", "the buff runs out"), they are the job
STATUS_WORDS = frozenset(('finished', 'done', 'complete', 'completed', 'failed', 'fails', 'succeeded', 'successful',
                          'successfully', 'worked', 'goes through', 'went through', 'runs out', 'ran out',
                          'wears off', 'expires', 'expired', 'fizzles out', 'went wrong', 'goes wrong', 'confirmed',
                          'approved', 'declined', 'rejected', 'denied'))
# two jobs that together mean a third: (a, b) -> c ("losing the buff" is a power-down; "the timer is done" a timer)
COMPOSE = {('lose', 'power_up'): 'power_down', ('lose', 'shield'): 'power_down', ('timer', 'success'): 'timer',
           ('lose', 'heal'): 'hurt', ('countdown', 'success'): 'countdown', ('cymbal', 'reverse'): 'reverse',
           # a stove turned on or its clicks are its igniter; a stream from a tap is the tap
           ('switch_on', 'ignite'): 'ignite', ('click', 'ignite'): 'ignite', ('stream', 'faucet'): 'faucet'}
# a concept that implies a broader one: the broader one is added at BROADER_WEIGHT
BROADER = {'switch_on': 'toggle', 'switch_off': 'toggle', 'light_switch': 'toggle', 'menu_select': 'menu',
           'menu_back': 'back', 'coin': 'pickup', 'gem': 'pickup', 'service_bell': 'bell', 'level_up': 'power_up',
           'one_up': 'power_up', 'receive': 'notify', 'cue': 'notify', 'thud': 'impact', 'clock': 'tick',
           'sub_drop': 'impact', 'braam': 'impact',
           'swish': 'whoosh', 'flyby': 'whoosh', 'coin_object': 'coin', 'glass_object': 'bell', 'countdown': 'timer',
           'kick_drum': 'drum', 'snare': 'drum', 'rimshot': 'drum', 'cross_stick': 'rimshot', 'tom': 'drum',
           'hihat': 'drum', 'cymbal': 'drum',
           'drum_roll': 'drum', 'sting': 'cue',
           # jobs no design plays by name: the gesture or sound that does them (a pull is a swipe, a copy a click)
           'refresh': 'swipe', 'copy': 'click'}
# a family whose name means something else in a request, or nothing: the job the family is for ('rise' and 'fall'
# are directions, a 'crash' family is the cymbal though "a crash" is a collision, the one 'drum' family the roll,
# a 'bed' the music under a video, not an ambience)
FAMILY_SENSE = {'rise': 'riser', 'fall': 'downlifter', 'crash': 'cymbal', 'ride': 'cymbal', 'kick': 'kick_drum',
                'drum': 'drum_roll', 'snap': 'finger_snap', 'sticks': 'count_in', 'bed': 'music',
                # recorded things (earshot.sounds.learned) whose family is the job by another word
                'step': 'footstep', 'switch': 'toggle', 'keyboard': 'type', 'camera': 'shutter', 'pass': 'flyby',
                'heart': 'heartbeat', 'dog': 'bark', 'car_horn': 'honk', 'baby': 'cry', 'phone': 'telephone', 
                'cat': 'meow', 'cow': 'moo', 'owl': 'hoot',
                'crow': 'caw', 'crowd': 'cheer', 'pan': 'sizzle', 'drink': 'pour', 'toilet': 'flush',
                'frog': 'croak', 'gun': 'gun_reload', 'spoon': 'stir', 'gas_stove': 'ignite'}
# a job that only one recorded thing does, where the thing's family also holds other sounds: {job: family}. A request
# for the job reaches the recordings of that family that list it ("chopping an onion": the knife on the board, not
# the knife's swish; "water boiling": the pot that boils, not the pot struck; brief.learned_ok)
JOB_THING = {'chop': 'knife', 'boil': 'pot'}
# a job of an app that has no sound of its own on a page: the job it is played as. A purchase or payment that went
# through is confirmed like any other success, with the same short click; the till's ring answers its own words only
# ("a cash register", "ka-ching": TILL_WORDS), a chime the words that ask for notes, and a game keeps its coins
PLAYED_AS = {'purchase': 'success'}
TILL_WORDS = frozenset(('cash register', 'ka-ching', 'kaching', 'cha-ching'))
# the tonal interface figures (earshot.sounds.ui): listeners heard them as unrecognisable, so a plain "tap",
# "hover", "toggle" or "next" gets a recorded switch, key or tap (earshot.sounds.learned, its `roles`); these answer
# when the request names them or asks for a tone ("a tonal tap": brief.BY_NAME)
BY_NAME_ONLY = frozenset(('tap/tone', 'hover/tone', 'toggle/up', 'toggle/down', 'next/step'))
# the interface jobs and the sound of a real thing that plays each (earshot.sounds.interface: a switch, a page, two
# knocks, a pop). A plain request for the job gets it; a pitched interface design (a figure on a marimba, a tick of a
# wood block) answers only a request that asks for music (brief.MUSIC_STEMS, an instrument named), and these then
# step back. tap and type stay with the recorded keys (earshot.sounds.learned `roles`)
INTERFACE_JOBS = {'hover': 'hover/fingertip', 'switch_on': 'toggle/key_down', 'switch_off': 'toggle/key_up',
                  'toggle': 'toggle/key_down', 'next': 'next/page', 'back': 'back/turn', 'undo': 'back/turn',
                  'open': 'open/drawer', 'close': 'close/drawer', 'send': 'send/swoosh', 'receive': 'receive/cork',
                  'notify': 'notify/knock', 'success': 'success/shutter', 'warning': 'warning/knock',
                  'error': 'error/knock', 'delete': 'delete/snip', 'like': 'like/pop'}
# plain requests of a film or a product video, said in these words and no others, and the design of real things that
# leads each (the film set's own sounds; brief.rank): a request with more words is ranked as ever
PLAIN_LEADS = {
    'a pop when an element appears': 'thump/door', 'something appears': 'thump/door', 'element appears': 'thump/door',
    'a tick': 'tick/key', 'a soft ui tick': 'tick/key', 'a small step': 'tick/key',
    'counter tick': 'tick/key', 'a cursor click': 'click/cursor', 'a card slide': 'slide/paper',
    'a panel slide': 'slide/paper', 'slide in': 'slide/paper', 'a fast whoosh': 'whoosh/fast',
    'whip pan': 'whoosh/fast', 'a soft whoosh': 'whoosh/sweep', 'a whoosh for a transition': 'whoosh/sweep',
    'a gentle whoosh': 'whoosh/sweep', 'a soft swish': 'whoosh/sweep', 'a title hit': 'title/bag',
    'a title slam': 'title/bag', 'a title card': 'title/bag', 'text slam': 'title/bag', 'an impact': 'impact/car',
    'a heavy impact': 'impact/car', 'a cinematic hit': 'impact/car', 'a logo hit': 'logo/bag',
    'a logo reveal hit': 'logo/bag', 'an end card hit': 'logo/bag', 'a logo reveal': 'logo/bag', 'a riser': 'rise/breath',
    'a riser before the drop': 'rise/breath', 'a build up': 'rise/breath', 'the drop': 'drop/sack',
    'a hit after the build': 'drop/sack', 'a shine': 'shine/blade', 'a glint': 'shine/blade',
    'a product shine': 'shine/blade', 'a tape stop': 'stop/zip', 'a pop': 'pop/mouth',
    # plain requests for a thing several recordings are of: the one heard most plainly
    'a hailstorm': 'hail/roof', 'hail': 'hail/roof', 'a reversing alarm': 'truck/reverse',
    'a reversing beep': 'truck/reverse', 'blowing': 'blow/puff'}
# where an interface job happens, not what it is: "open a menu" is an opening (brief.rank)
INTERFACE_PLACES = frozenset(('menu',))
# words that say which one of a thing or in which room it is, never what it belongs to: "the front door", "a kitchen
# timer" are a door and a timer (brief.owner_unknown); "a garage door" or "a car door" name another thing
WHICH_ONE = frozenset(('front', 'back', 'side', 'main', 'rear', 'top', 'bottom', 'inner', 'outer', 'upper', 'lower',
                       'old', 'new', 'big', 'little', 'small', 'large', 'kitchen', 'bathroom', 'bedroom', 'hall',
                       'hallway', 'cellar', 'basement', 'attic', 'office', 'closet', 'entrance'))
# the jobs of a motion film (a logo, a title hit, a transition, a riser, a glitch ...). Motion films use no
# instrument note, bell or synthesised tone as an effect: their effects are short low hits, thumps, clicks and
# passes of air, and their tonal content is their music. A request for one of these jobs (or one whose domain is a
# film or video) gets a sound of real things; a design with a note in it answers only when the words ask for music
# (brief.FILM_MUSIC) or the request is for a game (brief.rank)
MOTION_JOBS = frozenset(('logo', 'sting', 'impact', 'whoosh', 'swish', 'flyby', 'glitch', 'braam'))
# jobs that are motion jobs when the request's domain is a film or a video (an intro, something appearing, a
# shine on a product): in an app they keep their interface sounds
MOTION_DOMAIN_JOBS = frozenset(('startup', 'shimmer', 'pop', 'open', 'thud', 'riser', 'reverse', 'sub_drop',
                                'downlifter', 'text'))
# the word for the moment a film cuts everything and lands a hit (brief.read: in a film or video, not a fall)
MOTION_DROP = frozenset(('drop', 'drops'))
# an invented design -> the learned design (earshot.sounds.learned: recordings) of the same thing. An invented
# tactile sound scores 1 to 2 of 5 in listening tests and the recorded one 4 to 5, so a request that names the thing
# and reaches the invented one gets the recording (brief._preferred); the invented one still answers to its own id
TWINS = {'knock/door': 'door/knock', 'coins/handful': 'coins/floor', 'coin/drop': 'coins/floor',
         'glass/clink': 'glass/cheers',
         'paper/tear': 'paper/rip', 'paper/crumple': 'paper/ball', 'paper/page': 'page/turn', 'book/close': 'book/shut',
         'dice/roll': 'dice/throw', 'clock/ticktock': 'clock/mechanical', 'cork/pop': 'cork/bottle',
         'pen/click': 'pen/ballpoint', 'zipper/pull': 'zipper/jacket', 'scissors/snip': 'scissors/cut',
         'match/strike': 'match/light', 'stapler/press': 'stapler/desk', 'drawer/slide': 'drawer/wood',
         'door/close': 'door/shut', 'explosion/distant': 'explosion/far', 'hit/punch': 'punch/body',
         'swing/blade': 'sword/swing', 'splash/small': 'splash/plunge', 'keys/jingle': 'keys/bunch',
         'swish/soft': 'flick/thin', 'pop/soft': 'thump/appear', 'knock/desk': 'knock/table',
         'tap/table': 'tap/fingertip', 'bell/bike': 'bicycle_bell/ring', 'tape/peel': 'tape/roll',
         'water/pour': 'drink/pour', 'heartbeat/pulse': 'heart/beat', 'shot/pistol': 'gunshot/rifle',
         'click/soft': 'click/mouse', 'break/glass': 'glass/break'}
# words that ask for the action a person performs, not its single event (earshot.synth.perform; brief.performance):
# a walk, knocking, typing, a bunch of keys shaken, glasses clinked, a dog barking. A plural of the thing itself
# ("footsteps", "coins", "ice cubes") asks for it too; "a single", "one" and a bare singular ("a footstep", "a key
# press") give one event, since a game triggers one step per step
PERFORM_WORDS = frozenset((
    'walk', 'walks', 'walking', 'walked', 'footsteps', 'steps', 'stepping', 'pacing', 'strolling', 'marching',
    'running', 'treading', 'knocking', 'knocks', 'typing', 'typed', 'keystrokes', 'jingle', 'jingles', 'jingling',
    'jangle', 'jangling', 'rattling', 'shaking', 'shaken', 'clinking', 'clinks', 'clinked', 'toasting', 'clapping',
    'scattering', 'scattered', 'spilling', 'spilled', 'tumbling', 'clattering', 'barking', 'yapping'))
# phrases that ask for the action in so many words
PERFORM_PHRASES = ('on and off', 'off and on', 'a bunch of', 'a handful of', 'a burst of', 'a run of', 'over and over',
                   'again and again', 'in a row')
# a counted thing a person performs in a row ("four knocks on the table", "three claps", "two beeps"): one file of
# that many events, in this pattern of earshot.synth.perform (a countdown "3 2 1" is a `count`). Any other counted
# thing ("five coin pickups", "3 variations", "two versions") is that many separate files
ROW_WORDS = {'knock': 'knock', 'step': 'walk', 'footstep': 'walk', 'clap': 'row', 'tap': 'row', 'click': 'row',
             'beep': 'row', 'bleep': 'row', 'countdown': 'count', 'thump': 'row', 'thud': 'row', 'bump': 'row',
             'pop': 'row', 'tick': 'row', 'blip': 'row'}
# the action form of a row word -> the row word: "clapping", "knocking, 4 seconds" name the action, not one event
# (brief.acted). Applause is not one: it is a crowd (a recording of an audience, earshot.sounds.learned)
ACTION_WORDS = {'clapping': 'clap', 'knocking': 'knock', 'tapping': 'tap', 'clicking': 'click', 'beeping': 'beep',
                'bleeping': 'bleep', 'stepping': 'step', 'walking': 'step'}
# words that ask for a single event ("a single knock", "one step", "just one key press")
SINGLE_WORDS = ('a single', 'one single', 'single', 'just one', 'only one', 'one', 'a lone', 'an isolated')
# a twin that is the same thing only when the request says one of these words: the learned coins/floor is a handful
# of coins, so "a coin dropped on the table" keeps the invented single coin
TWIN_WORDS = {'coin/drop': ('coins', 'handful', 'change', 'pile'),
              'pop/soft': ('appear', 'appears', 'appearing', 'element', 'elements', 'screen', 'card', 'cards'),
              # a gun named, not a pistol: the recorded rifle shot (gunshot/rifle); an arrow "shot" or a pistol
              # keeps the invented one
              'shot/pistol': ('gunshot', 'gunshots', 'gun', 'guns', 'gunfire', 'rifle', 'firearm', 'shotgun'),
              # a click of the cursor or the mouse: the recorded mouse button; another soft click keeps its own
              'click/soft': ('cursor', 'cursors', 'mouse', 'pointer', 'trackpad')}
# a cymbal of the drum kit (earshot.sounds.drums) -> the film's cymbal (a learned crash cymbal struck hard, the
# material of earshot.sounds.motion): a film request ("a cymbal crash for the trailer") gets the film's, a request
# with a word of the kit ("a crash cymbal at the end of the fill") the kit's
FILM_CYMBAL = {'crash': 'cymbal/orchestral'}
KIT_WORDS = frozenset(('fill', 'fills', 'kit', 'beat', 'beats', 'drum', 'drums', 'drummer', 'groove', 'backbeat',
                       'band', 'song', 'track', 'loop', 'hi-hat', 'snare', 'kick', 'tom', 'toms'))
# the drums of the kit and the hand drums: a film or motion request without a word of the kit (KIT_WORDS) or the
# drum's own name gets none of them, its thud or hit is the film's own low hit
KIT_DRUMS = frozenset(('kick', 'tom', 'snare', 'cajon', 'conga', 'bongo', 'darbuka', 'frame_drum'))
# families that name a kind of sound, not a thing: a recorded one (whoosh/rope, knock/hollow) answers a request that
# also names its thing or character (brief.learned_ok), or plays the interface job it is chosen for (its roles)
SOUND_FAMILIES = frozenset(('whoosh', 'swish', 'swipe', 'click', 'tap', 'tick', 'knock', 'pop', 'thud', 'boom', 'hit',
                            'chime'))
# "the opposite of X", "like X but the other way"
OPPOSITE = {'riser': 'downlifter', 'switch_on': 'switch_off', 'open': 'close', 'next': 'back', 'send': 'receive',
            'power_up': 'power_down', 'win': 'lose', 'success': 'error', 'jump': 'land', 'unlock': 'lock',
            'startup': 'shutdown', 'add': 'delete'}
OPPOSITE.update({v: k for k, v in list(OPPOSITE.items())})

# which way the pitch or the gesture moves
DIRECTION = {'rising': 1, 'rise': 1, 'ascending': 1, 'ascend': 1, 'upward': 1, 'upwards': 1, 'climbing': 1,
             'climb': 1, 'lifting': 1, 'lift': 1, 'uplifting': 1, 'going up': 1, 'goes up': 1,
             'falling': -1, 'fall': -1, 'descending': -1, 'descend': -1, 'downward': -1, 'downwards': -1,
             'dropping': -1, 'drop': -1, 'sinking': -1, 'lowering': -1, 'going down': -1, 'goes down': -1,
             'winding down': -1, 'settling': -1, 'back up': 1}

# adjective -> {axis: delta}; the request's intensifiers scale it
MODIFIERS = {
    'soft': dict(hard=-0.5, energy=-0.3), 'gentle': dict(hard=-0.5, energy=-0.4), 'subtle': dict(energy=-0.6, hard=-0.3),
    'quiet': dict(energy=-0.6), 'delicate': dict(energy=-0.5, size=-0.3, hard=-0.3), 'smooth': dict(hard=-0.5),
    'mellow': dict(bright=-0.4, hard=-0.4), 'rounded': dict(hard=-0.5, bright=-0.2), 'round': dict(hard=-0.5, bright=-0.2),
    'calm': dict(energy=-0.5, motion=-0.3), 'relaxed': dict(energy=-0.5, motion=-0.3), 'light': dict(size=-0.4, energy=-0.2),
    'relaxing': dict(energy=-0.5, motion=-0.3), 'soothing': dict(energy=-0.5, hard=-0.4),
    'comforting': dict(energy=-0.3, bright=-0.2), 'reassuring': dict(energy=-0.3, hard=-0.3),
    'faint': dict(energy=-0.8), 'silent': dict(energy=-0.9), 'muted': dict(bright=-0.6, ring=-0.3),
    'hard': dict(hard=0.5), 'sharp': dict(hard=0.5, bright=0.3), 'snappy': dict(hard=0.5, ring=-0.4),
    'punchy': dict(hard=0.5, energy=0.4), 'aggressive': dict(energy=0.7, hard=0.5), 'intense': dict(energy=0.7),
    'powerful': dict(energy=0.6, size=0.4), 'strong': dict(energy=0.5), 'loud': dict(energy=0.6), 'bold': dict(energy=0.4, size=0.3),
    'firm': dict(hard=0.3, energy=0.2), 'solid': dict(size=0.3, hard=0.3), 'beefy': dict(size=0.6, energy=0.4),
    'chunky': dict(size=0.5, hard=0.3), 'thick': dict(size=0.4), 'crunchy': dict(hard=0.4, bright=0.3),
    'gritty': dict(bright=0.2, hard=0.3), 'exciting': dict(energy=0.5, motion=0.3),
    'bright': dict(bright=0.5), 'crisp': dict(bright=0.4, hard=0.3), 'shiny': dict(bright=0.5), 'sparkly': dict(bright=0.6),
    'airy': dict(bright=0.3, hard=-0.3), 'clear': dict(bright=0.3), 'clean': dict(bright=0.1), 'thin': dict(size=-0.5, bright=0.3),
    'dark': dict(bright=-0.6), 'warm': dict(bright=-0.4), 'muffled': dict(bright=-0.8), 'dull': dict(bright=-0.6),
    'deep': dict(size=0.6, bright=-0.3), 'low': dict(size=0.5, bright=-0.2), 'high': dict(size=-0.5, bright=0.2),
    'big': dict(size=0.6), 'large': dict(size=0.6), 'giant': dict(size=0.9, energy=0.4),
    'huge': dict(size=0.9, energy=0.4), 'massive': dict(size=1.0, energy=0.5), 'heavy': dict(size=0.7, hard=0.2),
    'epic': dict(size=0.8, space=0.5, energy=0.4), 'cinematic': dict(size=0.5, space=0.4), 'fat': dict(size=0.5),
    'thumpy': dict(size=0.4, hard=0.3),
    'small': dict(size=-0.5), 'tiny': dict(size=-0.9), 'little': dict(size=-0.5), 'mini': dict(size=-0.7), 'cute': dict(size=-0.5, bright=0.2, hard=-0.2),
    'long': dict(ring=0.6), 'ringing': dict(ring=0.7), 'sustained': dict(ring=0.7), 'lingering': dict(ring=0.6),
    'lush': dict(ring=0.4, width=0.3, space=0.3),
    'short': dict(ring=-0.6), 'quick': dict(ring=-0.5, motion=0.3), 'tight': dict(ring=-0.6, space=-0.4), 'brief': dict(ring=-0.6),
    'final': dict(ring=0.2, energy=0.2),
    'dry': dict(space=-1.0), 'intimate': dict(space=-0.5, energy=-0.3),
    'roomy': dict(space=0.5), 'spacious': dict(space=0.6, width=0.3), 'reverberant': dict(space=0.8), 'echoey': dict(space=0.8),
    'distant': dict(space=0.7, bright=-0.3, energy=-0.3), 'wet': dict(space=0.6), 'far': dict(space=0.7, energy=-0.3),
    'wide': dict(width=0.6), 'stereo': dict(width=0.4), 'mono': dict(width=-1.0), 'narrow': dict(width=-0.6),
    'fast': dict(motion=0.5), 'rapid': dict(motion=0.5), 'zippy': dict(motion=0.6, ring=-0.3),
    'energetic': dict(motion=0.5, energy=0.4), 'lively': dict(motion=0.4, energy=0.2),
    'bouncy': dict(motion=0.4, hard=-0.2),
    'swirling': dict(motion=0.5, width=0.3), 'swirly': dict(motion=0.5, width=0.3), 'warbling': dict(motion=0.5),
    'slow': dict(motion=-0.5, ring=0.2), 'steady': dict(motion=-0.5), 'lazy': dict(motion=-0.5),
    'floaty': dict(motion=-0.4, hard=-0.4), 'dreamy': dict(space=0.5, hard=-0.4),
    'ethereal': dict(space=0.6, bright=0.2),
    'angelic': dict(bright=0.3, space=0.4, hard=-0.3),
    'harsh': dict(bright=0.5, hard=0.5), 'piercing': dict(bright=0.7, hard=0.4), 'annoying': dict(bright=0.5, energy=0.5),
    'alarming': dict(bright=0.4, energy=0.5), 'dissonant': dict(bright=0.2), 'detuned': dict(motion=0.2),
    'premium': dict(hard=-0.2, energy=-0.2, space=0.1), 'elegant': dict(hard=-0.3, energy=-0.3), 'minimal': dict(energy=-0.4, ring=-0.2),
    'modern': dict(hard=-0.1), 'professional': dict(energy=-0.1), 'friendly': dict(hard=-0.3, bright=0.1),
    'simple': dict(energy=-0.2), 'neutral': dict(energy=-0.3),
    'playful': dict(size=-0.2, motion=0.3, bright=0.2), 'happy': dict(bright=0.3), 'cheerful': dict(bright=0.3, motion=0.2),
    'sad': dict(bright=-0.3, energy=-0.3), 'tense': dict(energy=0.3),
    'creepy': dict(bright=-0.5, size=0.4), 'unsettling': dict(bright=-0.4, size=0.3),
    'eerie': dict(bright=-0.4, space=0.4), 'spooky': dict(bright=-0.4, size=0.3), 'sinister': dict(bright=-0.6, size=0.5),
    'cosy': dict(bright=-0.4, energy=-0.3),
    'cozy': dict(bright=-0.4, energy=-0.3), 'positive': dict(bright=0.2), 'springy': dict(motion=0.4, hard=-0.2),
    'sweet': dict(bright=0.2, hard=-0.3), 'nostalgic': dict(bright=-0.2), 'bubbly': dict(motion=0.3, hard=-0.3),
    'rumbly': dict(size=0.6, bright=-0.4), 'menacing': dict(bright=-0.5, size=0.5), 'foreboding': dict(bright=-0.5, size=0.4),
    'serious': dict(bright=-0.2, size=0.2), 'ominous': dict(bright=-0.6, size=0.6), 'scary': dict(bright=-0.5, size=0.5),
    'booming': dict(size=0.6, energy=0.3),
    'dramatic': dict(size=0.5, energy=0.5, space=0.3), 'satisfying': dict(hard=0.1, size=0.1), 'tactile': dict(hard=0.2, space=-0.3),
    'reverb': dict(space=0.7), 'echo': dict(space=0.8), 'tail': dict(ring=0.5), 'room': dict(space=0.4),
    # the room a sound is heard in ("a scream in a small bathroom", "a shout in a stairwell"): how much of it
    'stairwell': dict(space=0.7), 'bathroom': dict(space=0.4),
}
INTENSIFIERS = {'very': 1.5, 'really': 1.4, 'super': 1.6, 'extremely': 1.8, 'ultra': 1.8, 'quite': 1.2, 'so': 1.3,
                'slightly': 0.5, 'slight': 0.5, 'somewhat': 0.6, 'little': 0.5, 'bit': 0.5, 'touch': 0.4,
                'kinda': 0.6, 'more': 1.0, 'much': 1.4, 'too': 1.0, 'almost': 0.7, 'barely': 0.3, 'pretty': 1.2,
                'fairly': 1.1, 'nice': 1.0, 'way': 1.5}
NEGATIONS = {'not': -0.7, 'no': -1.0, 'without': -1.0, 'less': -0.7, 'non': -0.7, 'never': -1.0, "isn't": -0.7,
             "don't": -0.7, "doesn't": -0.7, "shouldn't": -0.7, 'nothing': -1.0, 'avoid': -1.0, 'zero': -1.0,
             "won't": -0.7, 'wont': -0.7, "can't": -0.7, 'cannot': -0.7, "mustn't": -0.7, 'nor': -0.7,
             "wouldn't": -0.7, "aren't": -0.7, 'neither': -0.7}

# word -> material of taps, clicks and knocks / a hint for the voice's material (earshot.synth.voices.MATERIAL)
MATERIALS = {'wood': 'wood', 'wooden': 'wood', 'woody': 'wood', 'organic': 'wood', 'bamboo': 'wood',
             'glass': 'glass', 'glassy': 'glass', 'crystal': 'glass', 'crystalline': 'glass',
             'metal': 'metal', 'metallic': 'metal', 'steel': 'metal', 'brass': 'metal', 'tin': 'metal', 'iron': 'metal',
             'plastic': 'plastic', 'paper': 'paper', 'cardboard': 'paper', 'felt': 'soft', 'rubber': 'soft',
             'digital': 'digital', 'synthetic': 'digital', 'electronic': 'digital', 'synth': 'digital',
             'futuristic': 'digital', 'sci-fi': 'digital', 'scifi': 'digital', 'tech': 'digital', 'cyber': 'digital',
             'cyberpunk': 'digital', 'robot': 'digital', 'robotic': 'digital', 'analog': 'digital',
             'retro': 'retro', '8-bit': 'retro', '8bit': 'retro', '16-bit': 'retro', 'chiptune': 'retro',
             'chip': 'retro', 'arcade': 'retro', 'pixel': 'retro', 'nes': 'retro', 'snes': 'retro', 'gameboy': 'retro',
             'old-school': 'retro', 'oldschool': 'retro', 'square-wave': 'retro', 'cartridge': 'retro',
             'bitcrushed': 'retro', 'bottle': 'glass', 'sine': 'digital', 'sawtooth': 'digital', 'electro': 'digital',
             '808': 'digital',
             'nintendo': 'retro', 'atari': 'retro', 'commodore': 'retro', 'c64': 'retro', 'nes-style': 'retro',
             'mirror': 'glass', 'vase': 'glass', 'fence': 'wood', 'chair': 'wood', 'plank': 'wood', 'crate': 'wood',
             'barrel': 'wood', 'anvil': 'metal', 'helmet': 'metal', 'ceramic': 'glass', 'porcelain': 'glass'}
# phrases that name a material: the chip sound of old games, the waveforms of a synthesizer
MATERIAL_PHRASES = {'square wave': 'retro', 'old school': 'retro', 'old game': 'retro', 'old computer': 'retro',
                    'old console': 'retro', 'old video game': 'retro', 'pixel art': 'retro', 'game boy': 'retro',
                    'old cartridge': 'retro', 'old nes': 'retro', 'sine wave': 'digital', 'saw wave': 'digital',
                    'sawtooth wave': 'digital', 'drum machine': 'digital', 'beat machine': 'digital',
                    '808 kick': 'digital', 'kick 808': 'digital'}
# phrases that describe the envelope or the place, not a job: {phrase: {axis: delta}}
PHRASE_MODIFIERS = {'sharp attack': dict(hard=0.6), 'hard attack': dict(hard=0.6), 'fast attack': dict(hard=0.5),
                    'quick attack': dict(hard=0.5), 'soft attack': dict(hard=-0.6), 'slow attack': dict(hard=-0.7),
                    'long tail': dict(ring=0.6), 'short tail': dict(ring=-0.5), 'long decay': dict(ring=0.6),
                    'short decay': dict(ring=-0.5), 'quick decay': dict(ring=-0.5), 'long release': dict(ring=0.5),
                    'in the distance': dict(space=0.7, energy=-0.3), 'far away': dict(space=0.7, energy=-0.3),
                    'up close': dict(space=-0.6), 'close by': dict(space=-0.6),
                    'close-miked': dict(space=-0.6), 'close miked': dict(space=-0.6), 'close mic': dict(space=-0.6),
                    'close to the mic': dict(space=-0.6), 'close to the microphone': dict(space=-0.6)}
# a word whose job depends on the domain: (domain, word) -> concept, '' for none ("fails a stage" in a game is a
# loss; a "card" in an app is a component, not paper)
DOMAIN_SENSE = {('music', 'crash'): 'cymbal', ('game', 'kick'): 'hit',
                ('game', 'fail'): 'lose', ('game', 'failure'): 'lose', ('ui', 'sheet'): 'open',
                ('game', 'collapse'): 'break', ('motion', 'collapse'): 'break', ('ui', 'slide out'): 'open',
                ('ui', 'slide in'): 'open', ('game', 'activate'): 'power_up', ('game', 'jingle'): 'cue',
                ('game', 'reload'): 'gun_reload',
                ('game', 'enable'): 'power_up', ('ui', 'page'): '', ('ui', 'card'): '', ('game', 'star'): 'coin',
                ('game', 'stars'): 'coin', ('ui', 'item'): '', ('motion', 'horn'): 'braam',
                # what happens on a screen: a file picked up to drag it is a light tick; a title or logo that lands
                # or settles is a thud, bars that grow a riser, a wipe and a camera zoom moves of the film
                ('ui', 'pickup'): 'tick', ('ui', 'pick up'): 'tick', ('motion', 'land'): 'thud',
                ('motion', 'lands on'): 'thud', ('motion', 'grow'): 'riser', ('motion', 'bar'): '',
                ('motion', 'wipe'): 'whoosh', ('game', 'basket'): '',
                # a card or panel flipped over is a short swish; a logo that settles, comes to rest or locks up is
                # the logo's own low hit, not a falling sweep or a lock; in an app a thing grabbed is a light tick
                ('motion', 'flip'): 'swish', ('motion', 'settle into'): '', ('motion', 'come to rest'): '',
                ('motion', 'lock up'): '', ('ui', 'grab'): 'tick',
                # a card, a panel or a title that settles on a screen or in a film is a soft thump, as one that lands
                ('motion', 'settle'): 'thud', ('motion', 'settle into place'): 'thud', ('ui', 'settle'): 'thud',
                ('ui', 'settle into place'): 'thud', ('ui', 'settle into'): 'thud', ('ui', 'come to rest'): 'thud'}
# a job word that also says how the thing sounds: what settles or comes to rest does so softly (brief.read adds
# the adjective's axes and hints: the soft member of the job's family)
JOB_FEEL = {'settle': 'soft', 'settle into': 'soft', 'settle into place': 'soft', 'come to rest': 'soft'}
# which way a job moves, for designs whose own words do not say it
CONCEPT_DIR = {'riser': 1, 'downlifter': -1, 'power_up': 1, 'power_down': -1, 'level_up': 1, 'sub_drop': -1,
               'switch_on': 1, 'switch_off': -1, 'open': 1, 'close': -1, 'next': 1, 'back': -1, 'send': 1,
               'receive': -1, 'success': 1, 'error': -1, 'win': 1, 'lose': -1, 'jump': 1, 'land': -1,
               'startup': 1, 'shutdown': -1}
# words the stemmer would send to the wrong concept ('waving' is not 'waves', 'beating' eggs is no beat of music)
NOT_CONCEPT = frozenset(('waving', 'key', 'leaves', 'beating', 'dice', 'scratch', 'scratching',   # no 'scratched'
                         'dart', 'darts', 'blow'))    # the thing, a dart or a blow of breath; 'darting' is a dash
# function words of German, French, Spanish, Italian and Dutch that are no English words: a request that says one, or
# a word with a letter outside a to z that the lexicon does not know ("für"), is not written in English. In such a
# request these words and their English look-alikes (FOREIGN_ALIKE: German "die" is no death, French "pour" no
# pouring, "Regen" no healing) are read as nothing (brief.read)
FOREIGN_WORDS = frozenset((
    'für', 'der', 'das', 'den', 'dem', 'ein', 'eine', 'einen', 'einer', 'und', 'mit', 'wenn', 'beim', 'zum', 'zur',
    'vom', 'ist', 'nicht', 'auf', 'bei', 'oder', 'wird', 'ich', 'le', 'les', 'une', 'avec', 'dans', 'sur', 'et', 'du',
    'des', 'quand', 'est', 'el', 'los', 'las', 'para', 'con', 'por', 'una', 'del', 'cuando', 'il', 'gli', 'che',
    'het', 'een', 'voor', 'geluid', 'bruit', 'sonido', 'suono'))
FOREIGN_ALIKE = frozenset(('die', 'pour', 'sale', 'car', 'chat', 'pain', 'main', 'fin', 'bad', 'hell', 'fast', 'bald',
                           'toll', 'war', 'gift', 'hat', 'gut', 'once', 'pie', 'sin', 'tan', 'bin', 'regen', 'boot',
                           'lid', 'bug'))
# adjectives that are also a job's word: before a thing they say how it is done ("a careful step", "a cautious
# knock") and only soften it ({axis: delta}); alone they are the job ("a 'hey, careful' sound": a warning)
MANNER_WORDS = {'careful': dict(energy=-0.3, hard=-0.2), 'cautious': dict(energy=-0.3, hard=-0.2)}
# what follows these words is a description of the sound's envelope ("a shimmering decay"), not a job
ENVELOPE_WORDS = frozenset(('decay', 'tail', 'attack', 'release', 'ring', 'reverb', 'echo', 'finish'))
# verbs of a job whose object is the real thing asked for ("grabs a speed boost" is a power-up)
VERB_JOBS = frozenset(('pickup', 'swipe', 'swish'))
# jobs that act on something: what follows them is their object, a part of the sound ("parrying an attack")
OBJECT_VERBS = frozenset(('shield', 'dash', 'hover'))

# instruments: word or phrase -> earshot.synth.voices name
VOICES = {'piano': 'felt_piano', 'grand': 'grand_piano', 'grand piano': 'grand_piano', 'felt piano': 'felt_piano',
          'upright piano': 'felt_piano', 'kalimba': 'kalimba', 'mbira': 'kalimba', 'thumb piano': 'kalimba',
          'marimba': 'marimba', 'glockenspiel': 'glockenspiel', 'glock': 'glockenspiel',
          'vibraphone': 'vibraphone', 'vibes': 'vibraphone', 'celesta': 'celesta', 'xylophone': 'xylophone',
          'harp': 'harp', 'guitar': 'nylon_guitar', 'acoustic guitar': 'nylon_guitar', 'nylon guitar': 'nylon_guitar',
          'steel guitar': 'steel_guitar', 'rhodes': 'electric_piano', 'epiano': 'electric_piano',
          'e-piano': 'electric_piano', 'electric piano': 'electric_piano', 'handpan': 'handpan',
          'pizzicato': 'pizzicato', 'koto': 'koto', 'harpsichord': 'harpsichord', 'singing bowl': 'singing_bowl',
          'triangle': 'triangle', 'crotales': 'crotales', 'music box': 'music_box', 'musicbox': 'music_box',
          'lullaby': 'music_box', 'steel drum': 'steel_drum', 'steel pan': 'steel_drum', 'wine glass': 'wine_glass',
          'hand bell': 'hand_bell', 'handbell': 'hand_bell', 'church bell': 'church_bell',
          'tubular bell': 'tubular_bell', 'wind chime': 'wind_chime', 'upright bass': 'upright_bass',
          'double bass': 'upright_bass', 'bass': 'round_bass', 'sub bass': 'sub_bass', 'synth bass': 'pluck_bass',
          'reese': 'reese_bass', 'synth pluck': 'soft_pluck', 'plucky synth': 'soft_pluck',
          'electronic pluck': 'soft_pluck', 'electronic plink': 'soft_pluck', 'banjo': 'banjo',
          'dulcimer': 'dulcimer', 'cowbell': 'cowbell', 'wood block': 'wood_block', 'woodblock': 'wood_block',
          'wind-up': 'music_box', 'wind up': 'music_box', 'plucked string': 'harp', 'string pluck': 'harp',
          'low string': 'upright_bass', 'tine': 'kalimba', 'tines': 'kalimba', 'metal tine': 'kalimba',
          'wooden mallet': 'marimba', 'steel-string': 'steel_guitar',
          'plucked strings': 'pizzicato', 'pizz': 'pizzicato', 'strings plucked': 'pizzicato',
          'pizzicato strings': 'pizzicato', 'pizzicato string': 'pizzicato', 'string pizzicato': 'pizzicato',
          'steel string': 'steel_guitar', 'strum': 'nylon_guitar',
          'keyboard tone': 'felt_piano', 'keyboard note': 'felt_piano', 'ivory': 'grand_piano',
          'xylo': 'xylophone', 'cello pluck': 'pizzicato', 'violin pluck': 'pizzicato',
          'plucked violin': 'pizzicato', 'synth lead': 'square_lead', 'bass guitar': 'pluck_bass'}
# voices whose name a request also uses for the sound itself: the voice is then the head of the request
VOICE_FAMILIES = {'glock': 'glockenspiel', 'marimba': 'marimba', 'kalimba': 'kalimba', 'celesta': 'celesta'}

# word -> domain; DOMAIN_STRONG words decide over the others ("a browser game" is a game)
DOMAIN_WORDS = {'ui': 'ui', 'app': 'ui', 'website': 'ui', 'site': 'ui', 'web': 'ui', 'webapp': 'ui', 'page': 'ui',
                'landing': 'ui', 'dashboard': 'ui', 'interface': 'ui', 'saas': 'ui', 'mobile': 'ui', 'ios': 'ui',
                'android': 'ui', 'form': 'ui', 'desktop': 'ui', 'editor': 'ui', 'widget': 'ui', 'crm': 'ui',
                'banking': 'ui', 'fintech': 'ui', 'finance': 'ui', 'tracker': 'ui', 'calendar': 'ui', 'browser': 'ui',
                'portfolio': 'ui', 'checkout': 'ui', 'settings': 'ui', 'button': 'ui', 'tooltip': 'ui',
                'modal': 'ui', 'dialog': 'ui', 'onboarding': 'ui', 'screen': 'ui', 'email': 'ui', 'homepage': 'ui',
                'sidebar': 'ui', 'avatar': 'ui', 'chat': 'ui', 'toolbar': 'ui', 'navbar': 'ui', 'cart': 'ui',
                'feed': 'ui', 'profile': 'ui', 'login': 'ui', 'signup': 'ui', 
                'list': 'ui', 'wishlist': 'ui', 'carousel': 'ui', 'kanban': 'ui', 'sidebar': 'ui', 'toolbar': 'ui',
                'slider': 'ui', 'accordion': 'ui', 'clipboard': 'ui', 'icon': 'ui', 'nav': 'ui', 'checkbox': 'ui',
                'file': 'ui', 'folder': 'ui', 'document': 'ui', 'chart': 'motion', 'zoom': 'motion',
                'documentary': 'motion', 'teaser': 'motion', 'slideshow': 'motion',
                'sim': 'game', 'simulator': 'game', 'arena': 'game', 'brawler': 'game', 'mmorpg': 'game',
                'tactics': 'game', 'strategy': 'game', 'popup': 'ui', 'pop-up': 'ui', 'dropdown': 'ui',
                'drop-down': 'ui',
                'game': 'game', 'gaming': 'game', 'platformer': 'game', 'side-scroller': 'game', 'rpg': 'game',
                'mmo': 'game', 'shooter': 'game', 'shmup': 'game', 'roguelike': 'game', 'roguelite': 'game',
                'player': 'game', 'enemy': 'game', 'boss': 'game', 'level': 'game', 'arcade': 'game',
                'puzzle': 'game', 'match-3': 'game', 'character': 'game', 'npc': 'game', 'fps': 'game', 'kart': 'game',
                'racing': 'game', 'fighting': 'game', 'indie': 'game', 'quest': 'game', 'dungeon': 'game',
                'metroidvania': 'game', 'open-world': 'game', 'twin-stick': 'game', 'stage': 'game', 'hp': 'game',
                'video': 'motion', 'film': 'motion', 'trailer': 'motion', 'animation': 'motion', 'motion': 'motion',
                'promo': 'motion', 'reel': 'motion', 'tiktok': 'motion', 'explainer': 'motion', 'youtube': 'motion',
                'logo': 'motion', 'title': 'motion', 'cut': 'motion', 'scene': 'motion', 'keyframe': 'motion',
                'movie': 'motion', 'commercial': 'motion', 'graphics': 'motion', 'edit': 'motion', 'launch': 'motion',
                'ambience': 'ambience', 'ambient': 'ambience', 'background': 'ambience', 'atmosphere': 'ambience',
                'soundscape': 'ambience', 'loop': 'ambience',
                'music': 'music', 'melody': 'music', 'chord': 'music', 'note': 'music', 'jingle': 'music',
                'song': 'music', 'track': 'music', 'drum': 'music', 'drums': 'music', 'drummer': 'music',
                'kit': 'music', 'cymbal': 'music', 'cymbals': 'music', 'percussion': 'music', 'band': 'music'}
# phrases that only say where a sound plays: read as the domain, never as the paper, shot or screen in them
DOMAIN_PHRASES = {'title card': 'motion', 'end card': 'motion', 'lower third': 'motion', 'product shot': 'motion',
                  'hero shot': 'motion', 'title screen': 'game', 'home screen': 'ui', 'lock screen': 'ui',
                  'landing page': 'ui', 'hero section': 'ui', 'cut scene': 'game', 'cutscene': 'game',
                  'next shot': 'motion', 'two shots': 'motion', 'between shots': 'motion',
                  # a corner of the screen is where a product shows something
                  'bottom right': 'ui', 'bottom left': 'ui', 'top right': 'ui', 'top left': 'ui'}
# an enemy beaten is a moment of a game: the phrase says the domain as well as the blow
DOMAIN_PHRASES.update({p: 'game' for p in CONCEPTS['hit'] if 'enemy' in p.split()})
DOMAIN_STRONG = frozenset(('game', 'gaming', 'platformer', 'rpg', 'shooter', 'trailer', 'video', 'film', 'movie',
                           'app', 'website', 'dashboard'))

# words that carry no sound: dropped before matching
STOP = frozenset('a an the of for to in on at by with and or as is are be been being it its this that these those '
                 'when from into onto my our your their his her i we you me us he she they them sound sounds sfx '
                 'effect effects audio noise noises please need needs want wants like something anything everything '
                 'some kind sort type play plays played playing user users thing things stuff make makes made give '
                 'gives get gets getting got new good great cool awesome should would could will can have has had '
                 'while after before during each every one two three four few just someone somebody anyone what '
                 'which who whose how where there here then than but so also all any other else very hear heard '
                 'feel feels felt kinda sorta moment go goes going went come comes coming up down out off over '
                 'through across around away about under below above onto into back again only same such '
                 'feedback single "you\'d" "it\'s" "that\'s" "what\'s" "i\'d" let lets bit tell tells take takes '
                 'took look looks see way ones yeah hey oh ok-ish ish style styled themed generic realistic real '
                 'natural actual proper basic standard typical famous'.replace('"', '').split())

# kinds of sound no design is, by the word a request says for them: {word: the nearest design}. The word stays one
# the reader does not know, so the answer says "earshot makes no ..."; the table only chooses the nearest (a roar is
# a long low rumble, booing a crowd, a hiss a spray of air); brief.rank
SOUND_KINDS = {
    'roar': 'rumble/quake', 'roaring': 'rumble/quake', 'growl': 'rumble/quake', 'growling': 'rumble/quake',
    'snarl': 'rumble/quake', 'avalanche': 'rumble/quake', 'landslide': 'rumble/quake',
    'launch': 'rocket/launch', 'liftoff': 'rocket/launch', 'lift-off': 'rocket/launch',
    'blastoff': 'rocket/launch', 'takeoff': 'pass/jet', 'hiss': 'spray_can/spray', 'hissing': 'spray_can/spray',
    'fizz': 'spray_can/spray', 'fizzing': 'spray_can/spray', 'steam': 'spray_can/spray', 'squeak': 'door/creak',
    'squeaking': 'door/creak', 'squeaky': 'door/creak', 'squeal': 'tyres/screech', 'squealing': 'tyres/screech',
    'boo': 'crowd/boo', 'booing': 'crowd/boo', 'jeer': 'crowd/boo', 'jeering': 'crowd/boo',
    # a thing of lexicon.NOT_MADE or NOT_MADE_PHRASES and the design nearest to it (brief._nearest)
    'trumpet': 'win/fanfare', 'airlock': 'door/slide', 'fire alarm': 'timer/done',
    'smoke alarm': 'timer/done', 'car alarm': 'siren/pass', 'burglar alarm': 'siren/pass', 'toy gun': 'cap_gun/shot',
    'water gun': 'spray_can/spray', 'nerf gun': 'cap_gun/shot',
    'measuring tape': 'zip_tie/pull', 'tape measure': 'zip_tie/pull', 'pan set': 'pot_lid/set',
    'pan placed': 'pot_lid/set',
    'pan put down': 'pot_lid/set', 'saucepan set': 'pot_lid/set', 'saucepan placed': 'pot_lid/set',
    'saucepan put down': 'pot_lid/set',
    'moan': 'sigh/person', 'moaning': 'sigh/person', 'groan': 'sigh/person', 'groaning': 'sigh/person',
    'whisper': 'sigh/person', 'whispering': 'sigh/person', 'grunt': 'cough/person',
    'gulp': 'burp/person', 'slurp': 'straw/bubbles',
    'screeching': 'tyres/screech', 'flutter': 'wings/flap', 'fluttering': 'wings/flap',
    'whir': 'drill/electric', 'whirr': 'drill/electric', 'whirring': 'drill/electric',
    'sanding': 'broom/sweep', 'scrubbing': 'broom/sweep',
    'scraping': 'broom/sweep', 'brushing': 'broom/sweep', 'lightsaber': 'laser/beam', 'humming': 'space_hum/ship',
    'pecking': 'knock/wood', 'trumpeting': 'elephant/trumpet',
    'printer': 'paper/page', 'ladder': 'metal/drop',
    'machine gun': 'gunshot/rifle',
    'grunting': 'cough/person', 'gears': 'clock/mechanical',
    'clockwork': 'clock/mechanical', 'canoe': 'splash/small', 'paddling': 'splash/small'}
# the designs tried in this order when no word of a request found anything: a request always gets a nearest sound,
# said to be only that (brief.rank, make.doubt)
NEAREST_ANY = ('thump/soft', 'tap/soft', 'click/soft')
# the starting points offered then: three kinds of sound of the world, not three more thuds (make.starts)
STARTS_ANY = ('knock/wood', 'drill/electric', 'whoosh/soft')
# a surface, a ground or a place the catalogue has no design on, and the surfaces that sound nearest to it, nearest
# first ("rain on an umbrella" is rain on a tent, "rain on wood" rain on a roof, "footsteps in mud" steps on grass);
# brief.read adds them as words where the request says the surface after "on", "in", "against" or "into"
SURFACE_KIN = {
    'wood': ('floorboard', 'roof'), 'wooden': ('wood', 'floorboard', 'roof'), 'deck': ('wood', 'floorboard', 'roof'),
    'planks': ('wood', 'floorboard', 'roof'), 'porch': ('wood', 'floorboard', 'roof'), 'shed': ('roof',),
    'jetty': ('wood', 'floorboard'), 'pier': ('wood', 'floorboard'), 'boardwalk': ('wood', 'floorboard'),
    'umbrella': ('tent',), 'parasol': ('umbrella', 'tent'), 'tarp': ('tent',), 'tarpaulin': ('tent',),
    'canvas': ('tent',), 'awning': ('tent',), 'raincoat': ('umbrella', 'tent'), 'hood': ('umbrella', 'tent'),
    'mud': ('grass', 'dirt'), 'muddy': ('grass', 'dirt'),
    'swamp': ('grass', 'dirt'), 'moss': ('grass',), 'meadow': ('grass',), 'field': ('grass',),
    'asphalt': ('concrete', 'street'), 'pavement': ('concrete', 'street'), 'sidewalk': ('concrete', 'street'),
    'road': ('street', 'concrete'), 'tarmac': ('concrete', 'street'), 'cement': ('concrete',),
    'marble': ('tile', 'stone'), 'tiles': ('tile',), 'cobblestones': ('stone',), 'cobbles': ('stone',),
    'rock': ('stone', 'rocks'), 'sand': ('gravel',), 'beach': ('sand', 'gravel'), 'pebble': ('pebbles', 'gravel'),
    'parquet': ('floorboard', 'wood'), 'stairs': ('floorboard', 'wood'), 'staircase': ('floorboard', 'wood'),
    'ice': ('snow',), 'frost': ('snow',), 'pond': ('water',), 'lake': ('water',), 'sea': ('water',),
    'river': ('water',), 'pool': ('water',), 'puddle': ('water',), 'puddles': ('water',), 'bucket': ('water',),
    'car': ('tin', 'roof'), 'tin': ('roof',), 'gutter': ('tin', 'roof'), 'skylight': ('window',),
    'windshield': ('window',), 'windscreen': ('window',), 'pane': ('window',), 'windowpane': ('window',), 'trees': ('leaves',),
    'bushes': ('leaves',), 'foliage': ('leaves',), 'canopy': ('leaves',), 'grating': ('metal',), 'grate': ('metal',),
    'catwalk': ('metal',), 'pipe': ('metal',), 'steel': ('metal',), 'iron': ('metal',)}
# German words for the things, actions, materials and places the lexicon knows in English, lower case: a request
# that is not written in English is read through this table word by word ("Regen auf Holz": rain on wood;
# "Schritte auf Kies": footsteps on gravel), an inflected or compound word by its parts (brief.german)
GERMAN = {
    'regen': 'rain', 'regnet': 'rain', 'holz': 'wood', 'hölzern': 'wooden', 'holzboden': 'wooden floor',
    'schritt': 'footstep', 'schritte': 'footsteps', 'fußschritte': 'footsteps', 'laufen': 'footsteps',
    'gehen': 'footsteps', 'tür': 'door', 'türe': 'door', 'fenster': 'window', 'kies': 'gravel',
    'schotter': 'gravel', 'wind': 'wind', 'sturm': 'wind', 'feuer': 'fire', 'lagerfeuer': 'campfire',
    'wasser': 'water', 'glas': 'glass', 'gläser': 'glasses', 'metall': 'metal', 'blech': 'tin', 'stein': 'stone',
    'beton': 'concrete', 'schnee': 'snow', 'gras': 'grass', 'wiese': 'grass', 'laub': 'leaves',
    'blätter': 'leaves', 'dach': 'roof', 'zelt': 'tent', 'boden': 'floor', 'tisch': 'table',
    'schreibtisch': 'desk', 'klopfen': 'knock', 'klopft': 'knocking', 'anklopfen': 'knock', 'klick': 'click',
    'klicken': 'click', 'klickt': 'click', 'mausklick': 'mouse click', 'hund': 'dog', 'bellen': 'bark',
    'bellt': 'barking', 'gebell': 'bark', 'katze': 'cat', 'miauen': 'meow', 'miaut': 'meow',
    'schnurren': 'purr', 'schnurrt': 'purr', 'auto': 'car', 'motor': 'engine', 'hupe': 'car horn',
    'hupen': 'car horn', 'autotür': 'car door', 'schlüssel': 'keys', 'schlüsselbund': 'keys',
    'papier': 'paper', 'münze': 'coin', 'münzen': 'coins', 'erfolg': 'success', 'erfolgreich': 'success',
    'fehler': 'error', 'fehlermeldung': 'error', 'benachrichtigung': 'notification', 'nachricht': 'message',
    'warnung': 'warning', 'ton': 'sound', 'geräusch': 'sound', 'klang': 'sound', 'glocke': 'bell',
    'klingel': 'doorbell', 'türklingel': 'doorbell', 'klingeln': 'ring', 'klingelt': 'ringing',
    'telefon': 'telephone', 'handy': 'phone', 'tastatur': 'keyboard', 'tippen': 'typing', 'taste': 'key',
    'knopf': 'button', 'schalter': 'switch', 'lichtschalter': 'light switch', 'öffnen': 'open',
    'öffnet': 'opening', 'auf': 'on', 'schließen': 'close', 'schließt': 'closing', 'zu': 'shut',
    'zuschlagen': 'slam', 'knallt': 'slams', 'knallen': 'slam', 'knall': 'bang', 'schlag': 'hit',
    'schlagen': 'hit', 'schlägt': 'hitting', 'hammer': 'hammer', 'nagel': 'nail', 'säge': 'saw',
    'sägen': 'saw', 'bohrer': 'drill', 'bohrmaschine': 'drill', 'donner': 'thunder', 'gewitter': 'thunder',
    'welle': 'wave', 'wellen': 'waves', 'meer': 'waves', 'bach': 'stream', 'fluss': 'river', 'tropfen': 'drop',
    'tropft': 'dripping', 'wassertropfen': 'water drop', 'platschen': 'splash', 'spritzer': 'splash',
    'vogel': 'bird', 'vögel': 'birds', 'vogelgezwitscher': 'birds chirping', 'wald': 'forest',
    'nacht': 'night', 'grillen': 'crickets', 'frosch': 'frog', 'kuh': 'cow', 'pferd': 'horse', 'huhn': 'chicken',
    'hahn': 'rooster', 'schaf': 'sheep', 'schwein': 'pig', 'ente': 'duck', 'eule': 'owl', 'wolf': 'wolf',
    'herzschlag': 'heartbeat', 'herz': 'heartbeat', 'atmen': 'breath', 'husten': 'cough', 'niesen': 'sneeze',
    'lachen': 'laugh', 'gelächter': 'laughter', 'applaus': 'applause', 'klatschen': 'clapping',
    'menge': 'crowd', 'jubel': 'crowd cheering', 'kuss': 'kiss', 'schnarchen': 'snoring', 'gähnen': 'yawn',
    'schrei': 'scream', 'buch': 'book', 'seite': 'page', 'umblättern': 'page turn', 'stift': 'pen',
    'bleistift': 'pencil', 'schere': 'scissors', 'tasse': 'cup', 'teller': 'plate', 'besteck': 'cutlery',
    'flasche': 'bottle', 'korken': 'cork', 'messer': 'knife', 'topf': 'pot', 'pfanne': 'pan', 'löffel': 'spoon',
    'kühlschrank': 'fridge', 'uhr': 'clock', 'ticken': 'ticking', 'tickt': 'ticking', 'wecker': 'alarm',
    'kamera': 'camera', 'auslöser': 'shutter', 'schublade': 'drawer', 'reißverschluss': 'zipper',
    'zug': 'train', 'fahrrad': 'bicycle', 'fahrradklingel': 'bicycle bell', 'motorrad': 'motorcycle',
    'hubschrauber': 'helicopter', 'sirene': 'siren', 'schwert': 'sword', 'pfeil': 'arrow', 'schuss': 'gunshot',
    'explosion': 'explosion', 'laser': 'laser', 'zauber': 'magic spell', 'sprung': 'jump', 'springen': 'jump',
    'landen': 'land', 'gewonnen': 'win', 'verloren': 'lose', 'ball': 'ball', 'würfel': 'dice',
    'karten': 'cards', 'pfeife': 'whistle', 'eis': 'ice', 'eiswürfel': 'ice cubes', 'stoff': 'cloth',
    'leise': 'quiet', 'laut': 'loud', 'weich': 'soft', 'hart': 'hard', 'kurz': 'short', 'lang': 'long',
    'tief': 'deep', 'hell': 'bright', 'dunkel': 'dark', 'schnell': 'fast', 'langsam': 'slow', 'klein': 'small',
    'groß': 'big', 'schwer': 'heavy', 'leicht': 'light', 'warm': 'warm', 'sanft': 'gentle', 'weit': 'far',
    'entfernt': 'far away', 'nah': 'close', 'dreimal': 'three times', 'zweimal': 'twice', 'einmal': 'once',
    'sekunden': 'seconds', 'sekunde': 'second', 'fällt': 'falling', 'fallen': 'falling', 'fallend': 'falling',
    'rollen': 'rolling', 'bricht': 'breaking', 'zerbricht': 'breaking', 'zerbrechen': 'break',
    'gießen': 'pour', 'einschenken': 'pour', 'kochen': 'boiling', 'kocht': 'boiling', 'brutzeln': 'sizzle',
    'mit': 'with', 'und': 'and', 'für': 'for', 'gegen': 'against', 'im': 'in the', 'am': 'on the',
    'aus': 'from', 'hinter': 'behind', 'unter': 'under', 'über': 'above', 'durch': 'through', 'beim': 'when',
    'wenn': 'when', 'ohne': 'without', 'nicht': 'not', 'kein': 'no', 'keine': 'no', 'sehr': 'very',
    'ein': 'a', 'eine': 'a', 'einen': 'a', 'einem': 'a', 'einer': 'a', 'der': 'the', 'die': 'the', 'das': 'the',
    'den': 'the', 'dem': 'the', 'des': 'the', 'vom': 'from the', 'zum': 'to the', 'zur': 'to the', 'in': 'in',
    'von': 'of', 'wird': 'being', 'geöffnet': 'opened', 'geschlossen': 'closed', 'app': 'app', 'spiel': 'game',
    'hupt': 'honking', 'startet': 'starting', 'starten': 'start', 'fährt': 'goes', 'vorbei': 'passing',
    'ferne': 'distance', 'zauberspruch': 'magic spell', 'spruch': 'spell', 'wasserkocher': 'kettle',
    'spülmaschine': 'dishwasher', 'geschirrspüler': 'dishwasher',
    'kocher': 'kettle', 'kaffeemaschine': 'coffee machine', 'kaffee': 'coffee', 'maschine': 'machine',
    'tacker': 'stapler', 'hefter': 'stapler', 'niest': 'sneeze', 'jemand': 'someone', 'weinen': 'crying',
    'weint': 'crying', 'baby': 'baby', 'pfiff': 'whistle', 'schiedsrichter': 'referee', 'mischen': 'shuffling',
    'mischt': 'shuffling', 'zerreißen': 'tearing', 'zerreißt': 'tearing', 'reißen': 'tear', 'reißt': 'tearing',
    'hinweiston': 'notification', 'hinweis': 'notification', 'neu': 'new', 'nach': 'after', 'speichern': 'saving',
    'gespeichert': 'saved', 'übergang': 'transition', 'flur': 'hallway', 'publikum': 'audience', 'ziehen': 'draw',
    'zieht': 'drawing', 'einsammeln': 'pickup', 'sammeln': 'pickup', 'zwitschern': 'chirping',
    'zwitschert': 'chirping', 'wiehert': 'neighing', 'wiehern': 'neigh', 'pfeift': 'whistling', 'pfeifen': 'whistle',
    'strand': 'beach', 'schloss': 'lock', 'schlange': 'snake', 'schreibt': 'writing', 'schreiben': 'writing',
    'schneidet': 'cutting', 'schneiden': 'cut', 'prallt': 'bouncing', 'stadion': 'stadium', 'piepen': 'beep',
    'piept': 'beeping', 'piepton': 'beep', 'summen': 'buzz', 'summt': 'buzzing', 'rascheln': 'rustle',
    'raschelt': 'rustling', 'knistern': 'crackle', 'knistert': 'crackling', 'quietschen': 'squeak',
    'quietscht': 'squeaking', 'knarren': 'creak', 'knarrt': 'creaking', 'zischen': 'hiss', 'zischt': 'hissing',
    'brummen': 'hum', 'brummt': 'humming', 'blubbern': 'bubbling', 'gewehr': 'rifle', 'pistole': 'pistol',
    'faustschlag': 'punch', 'tritt': 'kick', 'truhe': 'chest', 'schatztruhe': 'treasure chest', 'heilen': 'heal',
    'heilung': 'healing', 'bogen': 'bow', 'schild': 'shield', 'drache': 'dragon', 'löwe': 'lion', 'brüllen': 'roar',
    'brüllt': 'roaring', 'heulen': 'howl', 'heult': 'howling', 'muhen': 'moo', 'krähe': 'crow', 'krähen': 'crowing',
    'gackern': 'clucking', 'quaken': 'quack', 'biene': 'bee', 'bienen': 'bees', 'fliege': 'fly', 'elefant': 'elephant',
    'ziege': 'goat', 'hagel': 'hail', 'blitz': 'lightning', 'brandung': 'waves', 'unterwasser': 'underwater',
    'blasen': 'bubbles', 'toilette': 'toilet', 'spülung': 'flush', 'wasserhahn': 'faucet', 'reifen': 'tyres',
    'flugzeug': 'airplane', 'staubsauger': 'vacuum cleaner', 'mikrowelle': 'microwave', 'besen': 'broom',
    'fegen': 'sweep', 'dose': 'can', 'schrank': 'cupboard', 'stuhl': 'chair', 'stempel': 'stamp',
    'schreibmaschine': 'typewriter', 'kasse': 'cash register', 'drucker': 'printer', 'maus': 'mouse',
    'atem': 'breath', 'seufzen': 'sigh', 'rülpsen': 'burp', 'schluckauf': 'hiccup', 'schreien': 'scream',
    'flüstern': 'whisper', 'schnipsen': 'finger snap', 'luftballon': 'balloon', 'ballon': 'balloon',
    'platzt': 'popping', 'platzen': 'pop', 'spielzeug': 'toy', 'treppe': 'stairs', 'rennen': 'running',
    'wasserfall': 'waterfall', 'kettensäge': 'chainsaw', 'rasenmäher': 'lawn mower', 'amboss': 'anvil',
    'feuerwerk': 'fireworks', 'kanone': 'cannon', 'granate': 'grenade', 'nachladen': 'reload', 'treffer': 'hit',
    'aufprall': 'impact', 'dumpf': 'dull', 'hohl': 'hollow', 'dunkle': 'dark', 'bei': 'at', 'ab': 'off'}
# a word of GERMAN that is English too and means something else there: it does not by itself mark a request as German
GERMAN_ALIKE = frozenset(('wind', 'hammer', 'laser', 'explosion', 'ball', 'wolf', 'auto', 'motor', 'in', 'warm',
                          'app', 'regen', 'ton', 'hell', 'lang', 'die', 'des', 'den', 'am', 'im', 'hart', 'glas',
                          'handy', 'taste', 'menge', 'bach', 'zug', 'stift', 'herz', 'aus', 'von', 'laut', 'weit',
                          'boden', 'kies', 'gras', 'spiel', 'tief', 'nah', 'zu', 'auf', 'mit', 'und', 'ein', 'der',
                          'fallen', 'teller', 'sprung', 'donner', 'klang', 'stein', 'blech', 'spritzer', 'klick',
                          'hinter', 'unter', 'das', 'dem', 'eis', 'uhr', 'buch', 'kurz', 'ente', 'eule', 'kuh',
                          'baby', 'strand', 'blitz', 'dose', 'ab'))
