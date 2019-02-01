
/bin/ls:     file format elf64-x86-64


Disassembly of section .init:

0000000000402168 <_init@@Base>:
  402168:	48 83 ec 08          	sub    $0x8,%rsp
  40216c:	48 8b 05 85 7e 21 00 	mov    0x217e85(%rip),%rax        # 619ff8 <__gmon_start__>
  402173:	48 85 c0             	test   %rax,%rax
  402176:	74 05                	je     40217d <_init@@Base+0x15>
  402178:	e8 33 04 00 00       	callq  4025b0 <__gmon_start__@plt>
  40217d:	48 83 c4 08          	add    $0x8,%rsp
  402181:	c3                   	retq   

Disassembly of section .plt:

0000000000402190 <__ctype_toupper_loc@plt-0x10>:
  402190:	ff 35 72 7e 21 00    	pushq  0x217e72(%rip)        # 61a008 <_fini@@Base+0x20810c>
  402196:	ff 25 74 7e 21 00    	jmpq   *0x217e74(%rip)        # 61a010 <_fini@@Base+0x208114>
  40219c:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004021a0 <__ctype_toupper_loc@plt>:
  4021a0:	ff 25 72 7e 21 00    	jmpq   *0x217e72(%rip)        # 61a018 <__ctype_toupper_loc@GLIBC_2.3>
  4021a6:	68 00 00 00 00       	pushq  $0x0
  4021ab:	e9 e0 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004021b0 <__uflow@plt>:
  4021b0:	ff 25 6a 7e 21 00    	jmpq   *0x217e6a(%rip)        # 61a020 <__uflow@GLIBC_2.2.5>
  4021b6:	68 01 00 00 00       	pushq  $0x1
  4021bb:	e9 d0 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004021c0 <getenv@plt>:
  4021c0:	ff 25 62 7e 21 00    	jmpq   *0x217e62(%rip)        # 61a028 <getenv@GLIBC_2.2.5>
  4021c6:	68 02 00 00 00       	pushq  $0x2
  4021cb:	e9 c0 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004021d0 <sigprocmask@plt>:
  4021d0:	ff 25 5a 7e 21 00    	jmpq   *0x217e5a(%rip)        # 61a030 <sigprocmask@GLIBC_2.2.5>
  4021d6:	68 03 00 00 00       	pushq  $0x3
  4021db:	e9 b0 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004021e0 <raise@plt>:
  4021e0:	ff 25 52 7e 21 00    	jmpq   *0x217e52(%rip)        # 61a038 <raise@GLIBC_2.2.5>
  4021e6:	68 04 00 00 00       	pushq  $0x4
  4021eb:	e9 a0 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004021f0 <free@plt>:
  4021f0:	ff 25 4a 7e 21 00    	jmpq   *0x217e4a(%rip)        # 61a040 <free@GLIBC_2.2.5>
  4021f6:	68 05 00 00 00       	pushq  $0x5
  4021fb:	e9 90 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402200 <localtime@plt>:
  402200:	ff 25 42 7e 21 00    	jmpq   *0x217e42(%rip)        # 61a048 <localtime@GLIBC_2.2.5>
  402206:	68 06 00 00 00       	pushq  $0x6
  40220b:	e9 80 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402210 <__mempcpy_chk@plt>:
  402210:	ff 25 3a 7e 21 00    	jmpq   *0x217e3a(%rip)        # 61a050 <__mempcpy_chk@GLIBC_2.3.4>
  402216:	68 07 00 00 00       	pushq  $0x7
  40221b:	e9 70 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402220 <abort@plt>:
  402220:	ff 25 32 7e 21 00    	jmpq   *0x217e32(%rip)        # 61a058 <abort@GLIBC_2.2.5>
  402226:	68 08 00 00 00       	pushq  $0x8
  40222b:	e9 60 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402230 <__errno_location@plt>:
  402230:	ff 25 2a 7e 21 00    	jmpq   *0x217e2a(%rip)        # 61a060 <__errno_location@GLIBC_2.2.5>
  402236:	68 09 00 00 00       	pushq  $0x9
  40223b:	e9 50 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402240 <strncmp@plt>:
  402240:	ff 25 22 7e 21 00    	jmpq   *0x217e22(%rip)        # 61a068 <strncmp@GLIBC_2.2.5>
  402246:	68 0a 00 00 00       	pushq  $0xa
  40224b:	e9 40 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402250 <_exit@plt>:
  402250:	ff 25 1a 7e 21 00    	jmpq   *0x217e1a(%rip)        # 61a070 <_exit@GLIBC_2.2.5>
  402256:	68 0b 00 00 00       	pushq  $0xb
  40225b:	e9 30 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402260 <strcpy@plt>:
  402260:	ff 25 12 7e 21 00    	jmpq   *0x217e12(%rip)        # 61a078 <strcpy@GLIBC_2.2.5>
  402266:	68 0c 00 00 00       	pushq  $0xc
  40226b:	e9 20 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402270 <__fpending@plt>:
  402270:	ff 25 0a 7e 21 00    	jmpq   *0x217e0a(%rip)        # 61a080 <__fpending@GLIBC_2.2.5>
  402276:	68 0d 00 00 00       	pushq  $0xd
  40227b:	e9 10 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402280 <isatty@plt>:
  402280:	ff 25 02 7e 21 00    	jmpq   *0x217e02(%rip)        # 61a088 <isatty@GLIBC_2.2.5>
  402286:	68 0e 00 00 00       	pushq  $0xe
  40228b:	e9 00 ff ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402290 <sigaction@plt>:
  402290:	ff 25 fa 7d 21 00    	jmpq   *0x217dfa(%rip)        # 61a090 <sigaction@GLIBC_2.2.5>
  402296:	68 0f 00 00 00       	pushq  $0xf
  40229b:	e9 f0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022a0 <iswcntrl@plt>:
  4022a0:	ff 25 f2 7d 21 00    	jmpq   *0x217df2(%rip)        # 61a098 <iswcntrl@GLIBC_2.2.5>
  4022a6:	68 10 00 00 00       	pushq  $0x10
  4022ab:	e9 e0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022b0 <wcswidth@plt>:
  4022b0:	ff 25 ea 7d 21 00    	jmpq   *0x217dea(%rip)        # 61a0a0 <wcswidth@GLIBC_2.2.5>
  4022b6:	68 11 00 00 00       	pushq  $0x11
  4022bb:	e9 d0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022c0 <localeconv@plt>:
  4022c0:	ff 25 e2 7d 21 00    	jmpq   *0x217de2(%rip)        # 61a0a8 <localeconv@GLIBC_2.2.5>
  4022c6:	68 12 00 00 00       	pushq  $0x12
  4022cb:	e9 c0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022d0 <mbstowcs@plt>:
  4022d0:	ff 25 da 7d 21 00    	jmpq   *0x217dda(%rip)        # 61a0b0 <mbstowcs@GLIBC_2.2.5>
  4022d6:	68 13 00 00 00       	pushq  $0x13
  4022db:	e9 b0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022e0 <readlink@plt>:
  4022e0:	ff 25 d2 7d 21 00    	jmpq   *0x217dd2(%rip)        # 61a0b8 <readlink@GLIBC_2.2.5>
  4022e6:	68 14 00 00 00       	pushq  $0x14
  4022eb:	e9 a0 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004022f0 <clock_gettime@plt>:
  4022f0:	ff 25 ca 7d 21 00    	jmpq   *0x217dca(%rip)        # 61a0c0 <clock_gettime@GLIBC_2.17>
  4022f6:	68 15 00 00 00       	pushq  $0x15
  4022fb:	e9 90 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402300 <textdomain@plt>:
  402300:	ff 25 c2 7d 21 00    	jmpq   *0x217dc2(%rip)        # 61a0c8 <textdomain@GLIBC_2.2.5>
  402306:	68 16 00 00 00       	pushq  $0x16
  40230b:	e9 80 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402310 <fclose@plt>:
  402310:	ff 25 ba 7d 21 00    	jmpq   *0x217dba(%rip)        # 61a0d0 <fclose@GLIBC_2.2.5>
  402316:	68 17 00 00 00       	pushq  $0x17
  40231b:	e9 70 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402320 <opendir@plt>:
  402320:	ff 25 b2 7d 21 00    	jmpq   *0x217db2(%rip)        # 61a0d8 <opendir@GLIBC_2.2.5>
  402326:	68 18 00 00 00       	pushq  $0x18
  40232b:	e9 60 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402330 <getpwuid@plt>:
  402330:	ff 25 aa 7d 21 00    	jmpq   *0x217daa(%rip)        # 61a0e0 <getpwuid@GLIBC_2.2.5>
  402336:	68 19 00 00 00       	pushq  $0x19
  40233b:	e9 50 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402340 <bindtextdomain@plt>:
  402340:	ff 25 a2 7d 21 00    	jmpq   *0x217da2(%rip)        # 61a0e8 <bindtextdomain@GLIBC_2.2.5>
  402346:	68 1a 00 00 00       	pushq  $0x1a
  40234b:	e9 40 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402350 <stpcpy@plt>:
  402350:	ff 25 9a 7d 21 00    	jmpq   *0x217d9a(%rip)        # 61a0f0 <stpcpy@GLIBC_2.2.5>
  402356:	68 1b 00 00 00       	pushq  $0x1b
  40235b:	e9 30 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402360 <dcgettext@plt>:
  402360:	ff 25 92 7d 21 00    	jmpq   *0x217d92(%rip)        # 61a0f8 <dcgettext@GLIBC_2.2.5>
  402366:	68 1c 00 00 00       	pushq  $0x1c
  40236b:	e9 20 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402370 <__ctype_get_mb_cur_max@plt>:
  402370:	ff 25 8a 7d 21 00    	jmpq   *0x217d8a(%rip)        # 61a100 <__ctype_get_mb_cur_max@GLIBC_2.2.5>
  402376:	68 1d 00 00 00       	pushq  $0x1d
  40237b:	e9 10 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402380 <strlen@plt>:
  402380:	ff 25 82 7d 21 00    	jmpq   *0x217d82(%rip)        # 61a108 <strlen@GLIBC_2.2.5>
  402386:	68 1e 00 00 00       	pushq  $0x1e
  40238b:	e9 00 fe ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402390 <__lxstat@plt>:
  402390:	ff 25 7a 7d 21 00    	jmpq   *0x217d7a(%rip)        # 61a110 <__lxstat@GLIBC_2.2.5>
  402396:	68 1f 00 00 00       	pushq  $0x1f
  40239b:	e9 f0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023a0 <__stack_chk_fail@plt>:
  4023a0:	ff 25 72 7d 21 00    	jmpq   *0x217d72(%rip)        # 61a118 <__stack_chk_fail@GLIBC_2.4>
  4023a6:	68 20 00 00 00       	pushq  $0x20
  4023ab:	e9 e0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023b0 <getopt_long@plt>:
  4023b0:	ff 25 6a 7d 21 00    	jmpq   *0x217d6a(%rip)        # 61a120 <getopt_long@GLIBC_2.2.5>
  4023b6:	68 21 00 00 00       	pushq  $0x21
  4023bb:	e9 d0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023c0 <mbrtowc@plt>:
  4023c0:	ff 25 62 7d 21 00    	jmpq   *0x217d62(%rip)        # 61a128 <mbrtowc@GLIBC_2.2.5>
  4023c6:	68 22 00 00 00       	pushq  $0x22
  4023cb:	e9 c0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023d0 <strchr@plt>:
  4023d0:	ff 25 5a 7d 21 00    	jmpq   *0x217d5a(%rip)        # 61a130 <strchr@GLIBC_2.2.5>
  4023d6:	68 23 00 00 00       	pushq  $0x23
  4023db:	e9 b0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023e0 <getgrgid@plt>:
  4023e0:	ff 25 52 7d 21 00    	jmpq   *0x217d52(%rip)        # 61a138 <getgrgid@GLIBC_2.2.5>
  4023e6:	68 24 00 00 00       	pushq  $0x24
  4023eb:	e9 a0 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004023f0 <_obstack_begin@plt>:
  4023f0:	ff 25 4a 7d 21 00    	jmpq   *0x217d4a(%rip)        # 61a140 <_obstack_begin@GLIBC_2.2.5>
  4023f6:	68 25 00 00 00       	pushq  $0x25
  4023fb:	e9 90 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402400 <__overflow@plt>:
  402400:	ff 25 42 7d 21 00    	jmpq   *0x217d42(%rip)        # 61a148 <__overflow@GLIBC_2.2.5>
  402406:	68 26 00 00 00       	pushq  $0x26
  40240b:	e9 80 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402410 <strrchr@plt>:
  402410:	ff 25 3a 7d 21 00    	jmpq   *0x217d3a(%rip)        # 61a150 <strrchr@GLIBC_2.2.5>
  402416:	68 27 00 00 00       	pushq  $0x27
  40241b:	e9 70 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402420 <fgetfilecon@plt>:
  402420:	ff 25 32 7d 21 00    	jmpq   *0x217d32(%rip)        # 61a158 <fgetfilecon>
  402426:	68 28 00 00 00       	pushq  $0x28
  40242b:	e9 60 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402430 <lseek@plt>:
  402430:	ff 25 2a 7d 21 00    	jmpq   *0x217d2a(%rip)        # 61a160 <lseek@GLIBC_2.2.5>
  402436:	68 29 00 00 00       	pushq  $0x29
  40243b:	e9 50 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402440 <gettimeofday@plt>:
  402440:	ff 25 22 7d 21 00    	jmpq   *0x217d22(%rip)        # 61a168 <gettimeofday@GLIBC_2.2.5>
  402446:	68 2a 00 00 00       	pushq  $0x2a
  40244b:	e9 40 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402450 <__assert_fail@plt>:
  402450:	ff 25 1a 7d 21 00    	jmpq   *0x217d1a(%rip)        # 61a170 <__assert_fail@GLIBC_2.2.5>
  402456:	68 2b 00 00 00       	pushq  $0x2b
  40245b:	e9 30 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402460 <__strtoul_internal@plt>:
  402460:	ff 25 12 7d 21 00    	jmpq   *0x217d12(%rip)        # 61a178 <__strtoul_internal@GLIBC_2.2.5>
  402466:	68 2c 00 00 00       	pushq  $0x2c
  40246b:	e9 20 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402470 <fnmatch@plt>:
  402470:	ff 25 0a 7d 21 00    	jmpq   *0x217d0a(%rip)        # 61a180 <fnmatch@GLIBC_2.2.5>
  402476:	68 2d 00 00 00       	pushq  $0x2d
  40247b:	e9 10 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402480 <memset@plt>:
  402480:	ff 25 02 7d 21 00    	jmpq   *0x217d02(%rip)        # 61a188 <memset@GLIBC_2.2.5>
  402486:	68 2e 00 00 00       	pushq  $0x2e
  40248b:	e9 00 fd ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402490 <acl_get_tag_type@plt>:
  402490:	ff 25 fa 7c 21 00    	jmpq   *0x217cfa(%rip)        # 61a190 <acl_get_tag_type@ACL_1.0>
  402496:	68 2f 00 00 00       	pushq  $0x2f
  40249b:	e9 f0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024a0 <fscanf@plt>:
  4024a0:	ff 25 f2 7c 21 00    	jmpq   *0x217cf2(%rip)        # 61a198 <fscanf@GLIBC_2.2.5>
  4024a6:	68 30 00 00 00       	pushq  $0x30
  4024ab:	e9 e0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024b0 <ioctl@plt>:
  4024b0:	ff 25 ea 7c 21 00    	jmpq   *0x217cea(%rip)        # 61a1a0 <ioctl@GLIBC_2.2.5>
  4024b6:	68 31 00 00 00       	pushq  $0x31
  4024bb:	e9 d0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024c0 <close@plt>:
  4024c0:	ff 25 e2 7c 21 00    	jmpq   *0x217ce2(%rip)        # 61a1a8 <close@GLIBC_2.2.5>
  4024c6:	68 32 00 00 00       	pushq  $0x32
  4024cb:	e9 c0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024d0 <acl_extended_file@plt>:
  4024d0:	ff 25 da 7c 21 00    	jmpq   *0x217cda(%rip)        # 61a1b0 <acl_extended_file@ACL_1.0>
  4024d6:	68 33 00 00 00       	pushq  $0x33
  4024db:	e9 b0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024e0 <closedir@plt>:
  4024e0:	ff 25 d2 7c 21 00    	jmpq   *0x217cd2(%rip)        # 61a1b8 <closedir@GLIBC_2.2.5>
  4024e6:	68 34 00 00 00       	pushq  $0x34
  4024eb:	e9 a0 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004024f0 <__libc_start_main@plt>:
  4024f0:	ff 25 ca 7c 21 00    	jmpq   *0x217cca(%rip)        # 61a1c0 <__libc_start_main@GLIBC_2.2.5>
  4024f6:	68 35 00 00 00       	pushq  $0x35
  4024fb:	e9 90 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402500 <memcmp@plt>:
  402500:	ff 25 c2 7c 21 00    	jmpq   *0x217cc2(%rip)        # 61a1c8 <memcmp@GLIBC_2.2.5>
  402506:	68 36 00 00 00       	pushq  $0x36
  40250b:	e9 80 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402510 <_setjmp@plt>:
  402510:	ff 25 ba 7c 21 00    	jmpq   *0x217cba(%rip)        # 61a1d0 <_setjmp@GLIBC_2.2.5>
  402516:	68 37 00 00 00       	pushq  $0x37
  40251b:	e9 70 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402520 <fputs_unlocked@plt>:
  402520:	ff 25 b2 7c 21 00    	jmpq   *0x217cb2(%rip)        # 61a1d8 <fputs_unlocked@GLIBC_2.2.5>
  402526:	68 38 00 00 00       	pushq  $0x38
  40252b:	e9 60 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402530 <calloc@plt>:
  402530:	ff 25 aa 7c 21 00    	jmpq   *0x217caa(%rip)        # 61a1e0 <calloc@GLIBC_2.2.5>
  402536:	68 39 00 00 00       	pushq  $0x39
  40253b:	e9 50 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402540 <lgetfilecon@plt>:
  402540:	ff 25 a2 7c 21 00    	jmpq   *0x217ca2(%rip)        # 61a1e8 <lgetfilecon>
  402546:	68 3a 00 00 00       	pushq  $0x3a
  40254b:	e9 40 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402550 <strcmp@plt>:
  402550:	ff 25 9a 7c 21 00    	jmpq   *0x217c9a(%rip)        # 61a1f0 <strcmp@GLIBC_2.2.5>
  402556:	68 3b 00 00 00       	pushq  $0x3b
  40255b:	e9 30 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402560 <signal@plt>:
  402560:	ff 25 92 7c 21 00    	jmpq   *0x217c92(%rip)        # 61a1f8 <signal@GLIBC_2.2.5>
  402566:	68 3c 00 00 00       	pushq  $0x3c
  40256b:	e9 20 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402570 <dirfd@plt>:
  402570:	ff 25 8a 7c 21 00    	jmpq   *0x217c8a(%rip)        # 61a200 <dirfd@GLIBC_2.2.5>
  402576:	68 3d 00 00 00       	pushq  $0x3d
  40257b:	e9 10 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402580 <getpwnam@plt>:
  402580:	ff 25 82 7c 21 00    	jmpq   *0x217c82(%rip)        # 61a208 <getpwnam@GLIBC_2.2.5>
  402586:	68 3e 00 00 00       	pushq  $0x3e
  40258b:	e9 00 fc ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402590 <__memcpy_chk@plt>:
  402590:	ff 25 7a 7c 21 00    	jmpq   *0x217c7a(%rip)        # 61a210 <__memcpy_chk@GLIBC_2.3.4>
  402596:	68 3f 00 00 00       	pushq  $0x3f
  40259b:	e9 f0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025a0 <sigemptyset@plt>:
  4025a0:	ff 25 72 7c 21 00    	jmpq   *0x217c72(%rip)        # 61a218 <sigemptyset@GLIBC_2.2.5>
  4025a6:	68 40 00 00 00       	pushq  $0x40
  4025ab:	e9 e0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025b0 <__gmon_start__@plt>:
  4025b0:	ff 25 6a 7c 21 00    	jmpq   *0x217c6a(%rip)        # 61a220 <__gmon_start__>
  4025b6:	68 41 00 00 00       	pushq  $0x41
  4025bb:	e9 d0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025c0 <memcpy@plt>:
  4025c0:	ff 25 62 7c 21 00    	jmpq   *0x217c62(%rip)        # 61a228 <memcpy@GLIBC_2.14>
  4025c6:	68 42 00 00 00       	pushq  $0x42
  4025cb:	e9 c0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025d0 <getgrnam@plt>:
  4025d0:	ff 25 5a 7c 21 00    	jmpq   *0x217c5a(%rip)        # 61a230 <getgrnam@GLIBC_2.2.5>
  4025d6:	68 43 00 00 00       	pushq  $0x43
  4025db:	e9 b0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025e0 <getfilecon@plt>:
  4025e0:	ff 25 52 7c 21 00    	jmpq   *0x217c52(%rip)        # 61a238 <getfilecon>
  4025e6:	68 44 00 00 00       	pushq  $0x44
  4025eb:	e9 a0 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004025f0 <fileno@plt>:
  4025f0:	ff 25 4a 7c 21 00    	jmpq   *0x217c4a(%rip)        # 61a240 <fileno@GLIBC_2.2.5>
  4025f6:	68 45 00 00 00       	pushq  $0x45
  4025fb:	e9 90 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402600 <tcgetpgrp@plt>:
  402600:	ff 25 42 7c 21 00    	jmpq   *0x217c42(%rip)        # 61a248 <tcgetpgrp@GLIBC_2.2.5>
  402606:	68 46 00 00 00       	pushq  $0x46
  40260b:	e9 80 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402610 <__xstat@plt>:
  402610:	ff 25 3a 7c 21 00    	jmpq   *0x217c3a(%rip)        # 61a250 <__xstat@GLIBC_2.2.5>
  402616:	68 47 00 00 00       	pushq  $0x47
  40261b:	e9 70 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402620 <readdir@plt>:
  402620:	ff 25 32 7c 21 00    	jmpq   *0x217c32(%rip)        # 61a258 <readdir@GLIBC_2.2.5>
  402626:	68 48 00 00 00       	pushq  $0x48
  40262b:	e9 60 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402630 <wcwidth@plt>:
  402630:	ff 25 2a 7c 21 00    	jmpq   *0x217c2a(%rip)        # 61a260 <wcwidth@GLIBC_2.2.5>
  402636:	68 49 00 00 00       	pushq  $0x49
  40263b:	e9 50 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402640 <malloc@plt>:
  402640:	ff 25 22 7c 21 00    	jmpq   *0x217c22(%rip)        # 61a268 <malloc@GLIBC_2.2.5>
  402646:	68 4a 00 00 00       	pushq  $0x4a
  40264b:	e9 40 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402650 <fflush@plt>:
  402650:	ff 25 1a 7c 21 00    	jmpq   *0x217c1a(%rip)        # 61a270 <fflush@GLIBC_2.2.5>
  402656:	68 4b 00 00 00       	pushq  $0x4b
  40265b:	e9 30 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402660 <nl_langinfo@plt>:
  402660:	ff 25 12 7c 21 00    	jmpq   *0x217c12(%rip)        # 61a278 <nl_langinfo@GLIBC_2.2.5>
  402666:	68 4c 00 00 00       	pushq  $0x4c
  40266b:	e9 20 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402670 <ungetc@plt>:
  402670:	ff 25 0a 7c 21 00    	jmpq   *0x217c0a(%rip)        # 61a280 <ungetc@GLIBC_2.2.5>
  402676:	68 4d 00 00 00       	pushq  $0x4d
  40267b:	e9 10 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402680 <__fxstat@plt>:
  402680:	ff 25 02 7c 21 00    	jmpq   *0x217c02(%rip)        # 61a288 <__fxstat@GLIBC_2.2.5>
  402686:	68 4e 00 00 00       	pushq  $0x4e
  40268b:	e9 00 fb ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402690 <strcoll@plt>:
  402690:	ff 25 fa 7b 21 00    	jmpq   *0x217bfa(%rip)        # 61a290 <strcoll@GLIBC_2.2.5>
  402696:	68 4f 00 00 00       	pushq  $0x4f
  40269b:	e9 f0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026a0 <mktime@plt>:
  4026a0:	ff 25 f2 7b 21 00    	jmpq   *0x217bf2(%rip)        # 61a298 <mktime@GLIBC_2.2.5>
  4026a6:	68 50 00 00 00       	pushq  $0x50
  4026ab:	e9 e0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026b0 <__freading@plt>:
  4026b0:	ff 25 ea 7b 21 00    	jmpq   *0x217bea(%rip)        # 61a2a0 <__freading@GLIBC_2.2.5>
  4026b6:	68 51 00 00 00       	pushq  $0x51
  4026bb:	e9 d0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026c0 <fwrite_unlocked@plt>:
  4026c0:	ff 25 e2 7b 21 00    	jmpq   *0x217be2(%rip)        # 61a2a8 <fwrite_unlocked@GLIBC_2.2.5>
  4026c6:	68 52 00 00 00       	pushq  $0x52
  4026cb:	e9 c0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026d0 <acl_get_entry@plt>:
  4026d0:	ff 25 da 7b 21 00    	jmpq   *0x217bda(%rip)        # 61a2b0 <acl_get_entry@ACL_1.0>
  4026d6:	68 53 00 00 00       	pushq  $0x53
  4026db:	e9 b0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026e0 <realloc@plt>:
  4026e0:	ff 25 d2 7b 21 00    	jmpq   *0x217bd2(%rip)        # 61a2b8 <realloc@GLIBC_2.2.5>
  4026e6:	68 54 00 00 00       	pushq  $0x54
  4026eb:	e9 a0 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004026f0 <stpncpy@plt>:
  4026f0:	ff 25 ca 7b 21 00    	jmpq   *0x217bca(%rip)        # 61a2c0 <stpncpy@GLIBC_2.2.5>
  4026f6:	68 55 00 00 00       	pushq  $0x55
  4026fb:	e9 90 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402700 <fdopen@plt>:
  402700:	ff 25 c2 7b 21 00    	jmpq   *0x217bc2(%rip)        # 61a2c8 <fdopen@GLIBC_2.2.5>
  402706:	68 56 00 00 00       	pushq  $0x56
  40270b:	e9 80 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402710 <setlocale@plt>:
  402710:	ff 25 ba 7b 21 00    	jmpq   *0x217bba(%rip)        # 61a2d0 <setlocale@GLIBC_2.2.5>
  402716:	68 57 00 00 00       	pushq  $0x57
  40271b:	e9 70 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402720 <_obstack_newchunk@plt>:
  402720:	ff 25 b2 7b 21 00    	jmpq   *0x217bb2(%rip)        # 61a2d8 <_obstack_newchunk@GLIBC_2.2.5>
  402726:	68 58 00 00 00       	pushq  $0x58
  40272b:	e9 60 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402730 <__printf_chk@plt>:
  402730:	ff 25 aa 7b 21 00    	jmpq   *0x217baa(%rip)        # 61a2e0 <__printf_chk@GLIBC_2.3.4>
  402736:	68 59 00 00 00       	pushq  $0x59
  40273b:	e9 50 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402740 <strftime@plt>:
  402740:	ff 25 a2 7b 21 00    	jmpq   *0x217ba2(%rip)        # 61a2e8 <strftime@GLIBC_2.2.5>
  402746:	68 5a 00 00 00       	pushq  $0x5a
  40274b:	e9 40 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402750 <mempcpy@plt>:
  402750:	ff 25 9a 7b 21 00    	jmpq   *0x217b9a(%rip)        # 61a2f0 <mempcpy@GLIBC_2.2.5>
  402756:	68 5b 00 00 00       	pushq  $0x5b
  40275b:	e9 30 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402760 <memmove@plt>:
  402760:	ff 25 92 7b 21 00    	jmpq   *0x217b92(%rip)        # 61a2f8 <memmove@GLIBC_2.2.5>
  402766:	68 5c 00 00 00       	pushq  $0x5c
  40276b:	e9 20 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402770 <error@plt>:
  402770:	ff 25 8a 7b 21 00    	jmpq   *0x217b8a(%rip)        # 61a300 <error@GLIBC_2.2.5>
  402776:	68 5d 00 00 00       	pushq  $0x5d
  40277b:	e9 10 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402780 <open@plt>:
  402780:	ff 25 82 7b 21 00    	jmpq   *0x217b82(%rip)        # 61a308 <open@GLIBC_2.2.5>
  402786:	68 5e 00 00 00       	pushq  $0x5e
  40278b:	e9 00 fa ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402790 <fseeko@plt>:
  402790:	ff 25 7a 7b 21 00    	jmpq   *0x217b7a(%rip)        # 61a310 <fseeko@GLIBC_2.2.5>
  402796:	68 5f 00 00 00       	pushq  $0x5f
  40279b:	e9 f0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027a0 <strtoul@plt>:
  4027a0:	ff 25 72 7b 21 00    	jmpq   *0x217b72(%rip)        # 61a318 <strtoul@GLIBC_2.2.5>
  4027a6:	68 60 00 00 00       	pushq  $0x60
  4027ab:	e9 e0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027b0 <__cxa_atexit@plt>:
  4027b0:	ff 25 6a 7b 21 00    	jmpq   *0x217b6a(%rip)        # 61a320 <__cxa_atexit@GLIBC_2.2.5>
  4027b6:	68 61 00 00 00       	pushq  $0x61
  4027bb:	e9 d0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027c0 <wcstombs@plt>:
  4027c0:	ff 25 62 7b 21 00    	jmpq   *0x217b62(%rip)        # 61a328 <wcstombs@GLIBC_2.2.5>
  4027c6:	68 62 00 00 00       	pushq  $0x62
  4027cb:	e9 c0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027d0 <freecon@plt>:
  4027d0:	ff 25 5a 7b 21 00    	jmpq   *0x217b5a(%rip)        # 61a330 <freecon>
  4027d6:	68 63 00 00 00       	pushq  $0x63
  4027db:	e9 b0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027e0 <sigismember@plt>:
  4027e0:	ff 25 52 7b 21 00    	jmpq   *0x217b52(%rip)        # 61a338 <sigismember@GLIBC_2.2.5>
  4027e6:	68 64 00 00 00       	pushq  $0x64
  4027eb:	e9 a0 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

00000000004027f0 <exit@plt>:
  4027f0:	ff 25 4a 7b 21 00    	jmpq   *0x217b4a(%rip)        # 61a340 <exit@GLIBC_2.2.5>
  4027f6:	68 65 00 00 00       	pushq  $0x65
  4027fb:	e9 90 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402800 <fwrite@plt>:
  402800:	ff 25 42 7b 21 00    	jmpq   *0x217b42(%rip)        # 61a348 <fwrite@GLIBC_2.2.5>
  402806:	68 66 00 00 00       	pushq  $0x66
  40280b:	e9 80 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402810 <__fprintf_chk@plt>:
  402810:	ff 25 3a 7b 21 00    	jmpq   *0x217b3a(%rip)        # 61a350 <__fprintf_chk@GLIBC_2.3.4>
  402816:	68 67 00 00 00       	pushq  $0x67
  40281b:	e9 70 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402820 <fflush_unlocked@plt>:
  402820:	ff 25 32 7b 21 00    	jmpq   *0x217b32(%rip)        # 61a358 <fflush_unlocked@GLIBC_2.2.5>
  402826:	68 68 00 00 00       	pushq  $0x68
  40282b:	e9 60 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402830 <mbsinit@plt>:
  402830:	ff 25 2a 7b 21 00    	jmpq   *0x217b2a(%rip)        # 61a360 <mbsinit@GLIBC_2.2.5>
  402836:	68 69 00 00 00       	pushq  $0x69
  40283b:	e9 50 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402840 <iswprint@plt>:
  402840:	ff 25 22 7b 21 00    	jmpq   *0x217b22(%rip)        # 61a368 <iswprint@GLIBC_2.2.5>
  402846:	68 6a 00 00 00       	pushq  $0x6a
  40284b:	e9 40 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402850 <sigaddset@plt>:
  402850:	ff 25 1a 7b 21 00    	jmpq   *0x217b1a(%rip)        # 61a370 <sigaddset@GLIBC_2.2.5>
  402856:	68 6b 00 00 00       	pushq  $0x6b
  40285b:	e9 30 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402860 <strstr@plt>:
  402860:	ff 25 12 7b 21 00    	jmpq   *0x217b12(%rip)        # 61a378 <strstr@GLIBC_2.2.5>
  402866:	68 6c 00 00 00       	pushq  $0x6c
  40286b:	e9 20 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402870 <__ctype_tolower_loc@plt>:
  402870:	ff 25 0a 7b 21 00    	jmpq   *0x217b0a(%rip)        # 61a380 <__ctype_tolower_loc@GLIBC_2.3>
  402876:	68 6d 00 00 00       	pushq  $0x6d
  40287b:	e9 10 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402880 <__ctype_b_loc@plt>:
  402880:	ff 25 02 7b 21 00    	jmpq   *0x217b02(%rip)        # 61a388 <__ctype_b_loc@GLIBC_2.3>
  402886:	68 6e 00 00 00       	pushq  $0x6e
  40288b:	e9 00 f9 ff ff       	jmpq   402190 <_init@@Base+0x28>

0000000000402890 <__sprintf_chk@plt>:
  402890:	ff 25 fa 7a 21 00    	jmpq   *0x217afa(%rip)        # 61a390 <__sprintf_chk@GLIBC_2.3.4>
  402896:	68 6f 00 00 00       	pushq  $0x6f
  40289b:	e9 f0 f8 ff ff       	jmpq   402190 <_init@@Base+0x28>

Disassembly of section .text:

00000000004028a0 <.text>:
  4028a0:	50                   	push   %rax
  4028a1:	b9 88 2c 41 00       	mov    $0x412c88,%ecx
  4028a6:	ba a6 0e 00 00       	mov    $0xea6,%edx
  4028ab:	be 36 37 41 00       	mov    $0x413736,%esi
  4028b0:	bf 98 3c 41 00       	mov    $0x413c98,%edi
  4028b5:	e8 96 fb ff ff       	callq  402450 <__assert_fail@plt>
  4028ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4028c0:	41 57                	push   %r15
  4028c2:	41 56                	push   %r14
  4028c4:	41 55                	push   %r13
  4028c6:	41 54                	push   %r12
  4028c8:	55                   	push   %rbp
  4028c9:	48 89 f5             	mov    %rsi,%rbp
  4028cc:	53                   	push   %rbx
  4028cd:	89 fb                	mov    %edi,%ebx
  4028cf:	48 81 ec 88 03 00 00 	sub    $0x388,%rsp
  4028d6:	48 8b 3e             	mov    (%rsi),%rdi
  4028d9:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4028e0:	00 00 
  4028e2:	48 89 84 24 78 03 00 	mov    %rax,0x378(%rsp)
  4028e9:	00 
  4028ea:	31 c0                	xor    %eax,%eax
  4028ec:	e8 af ad 00 00       	callq  40d6a0 <__sprintf_chk@plt+0xae10>
  4028f1:	be 19 69 41 00       	mov    $0x416919,%esi
  4028f6:	bf 06 00 00 00       	mov    $0x6,%edi
  4028fb:	e8 10 fe ff ff       	callq  402710 <setlocale@plt>
  402900:	be 1c 38 41 00       	mov    $0x41381c,%esi
  402905:	bf 00 38 41 00       	mov    $0x413800,%edi
  40290a:	e8 31 fa ff ff       	callq  402340 <bindtextdomain@plt>
  40290f:	bf 00 38 41 00       	mov    $0x413800,%edi
  402914:	e8 e7 f9 ff ff       	callq  402300 <textdomain@plt>
  402919:	bf 00 a2 40 00       	mov    $0x40a200,%edi
  40291e:	c7 05 58 7c 21 00 02 	movl   $0x2,0x217c58(%rip)        # 61a580 <_fini@@Base+0x208684>
  402925:	00 00 00 
  402928:	e8 b3 f5 00 00       	callq  411ee0 <__sprintf_chk@plt+0xf650>
  40292d:	48 b8 00 00 00 00 00 	movabs $0x8000000000000000,%rax
  402934:	00 00 80 
  402937:	c7 05 ef 86 21 00 00 	movl   $0x0,0x2186ef(%rip)        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  40293e:	00 00 00 
  402941:	c6 05 88 87 21 00 01 	movb   $0x1,0x218788(%rip)        # 61b0d0 <stderr@@GLIBC_2.2.5+0xa80>
  402948:	48 89 05 31 88 21 00 	mov    %rax,0x218831(%rip)        # 61b180 <stderr@@GLIBC_2.2.5+0xb30>
  40294f:	8b 05 17 7c 21 00    	mov    0x217c17(%rip),%eax        # 61a56c <_fini@@Base+0x208670>
  402955:	48 c7 05 30 88 21 00 	movq   $0x0,0x218830(%rip)        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  40295c:	00 00 00 00 
  402960:	48 c7 05 1d 88 21 00 	movq   $0xffffffffffffffff,0x21881d(%rip)        # 61b188 <stderr@@GLIBC_2.2.5+0xb38>
  402967:	ff ff ff ff 
  40296b:	c6 05 7e 87 21 00 00 	movb   $0x0,0x21877e(%rip)        # 61b0f0 <stderr@@GLIBC_2.2.5+0xaa0>
  402972:	83 f8 02             	cmp    $0x2,%eax
  402975:	0f 84 88 08 00 00    	je     403203 <__sprintf_chk@plt+0x973>
  40297b:	83 f8 03             	cmp    $0x3,%eax
  40297e:	74 2f                	je     4029af <__sprintf_chk@plt+0x11f>
  402980:	83 e8 01             	sub    $0x1,%eax
  402983:	74 05                	je     40298a <__sprintf_chk@plt+0xfa>
  402985:	e8 96 f8 ff ff       	callq  402220 <abort@plt>
  40298a:	bf 01 00 00 00       	mov    $0x1,%edi
  40298f:	e8 ec f8 ff ff       	callq  402280 <isatty@plt>
  402994:	85 c0                	test   %eax,%eax
  402996:	0f 84 50 0e 00 00    	je     4037ec <__sprintf_chk@plt+0xf5c>
  40299c:	c7 05 aa 87 21 00 02 	movl   $0x2,0x2187aa(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4029a3:	00 00 00 
  4029a6:	c6 05 43 87 21 00 01 	movb   $0x1,0x218743(%rip)        # 61b0f0 <stderr@@GLIBC_2.2.5+0xaa0>
  4029ad:	eb 16                	jmp    4029c5 <__sprintf_chk@plt+0x135>
  4029af:	be 05 00 00 00       	mov    $0x5,%esi
  4029b4:	31 ff                	xor    %edi,%edi
  4029b6:	c7 05 90 87 21 00 00 	movl   $0x0,0x218790(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4029bd:	00 00 00 
  4029c0:	e8 7b bc 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  4029c5:	bf 2e 38 41 00       	mov    $0x41382e,%edi
  4029ca:	c7 05 78 87 21 00 00 	movl   $0x0,0x218778(%rip)        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  4029d1:	00 00 00 
  4029d4:	c7 05 6a 87 21 00 00 	movl   $0x0,0x21876a(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  4029db:	00 00 00 
  4029de:	c6 05 62 87 21 00 00 	movb   $0x0,0x218762(%rip)        # 61b147 <stderr@@GLIBC_2.2.5+0xaf7>
  4029e5:	c6 05 59 87 21 00 00 	movb   $0x0,0x218759(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  4029ec:	c6 05 51 87 21 00 00 	movb   $0x0,0x218751(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  4029f3:	c7 05 2f 87 21 00 00 	movl   $0x0,0x21872f(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  4029fa:	00 00 00 
  4029fd:	c6 05 10 87 21 00 00 	movb   $0x0,0x218710(%rip)        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  402a04:	c7 05 02 87 21 00 01 	movl   $0x1,0x218702(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  402a0b:	00 00 00 
  402a0e:	c6 05 f9 86 21 00 00 	movb   $0x0,0x2186f9(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  402a15:	c6 05 f1 86 21 00 00 	movb   $0x0,0x2186f1(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  402a1c:	c7 05 e2 86 21 00 00 	movl   $0x0,0x2186e2(%rip)        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  402a23:	00 00 00 
  402a26:	48 c7 05 cf 86 21 00 	movq   $0x0,0x2186cf(%rip)        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402a2d:	00 00 00 00 
  402a31:	48 c7 05 bc 86 21 00 	movq   $0x0,0x2186bc(%rip)        # 61b0f8 <stderr@@GLIBC_2.2.5+0xaa8>
  402a38:	00 00 00 00 
  402a3c:	c6 05 3a 87 21 00 00 	movb   $0x0,0x21873a(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  402a43:	e8 78 f7 ff ff       	callq  4021c0 <getenv@plt>
  402a48:	48 85 c0             	test   %rax,%rax
  402a4b:	49 89 c4             	mov    %rax,%r12
  402a4e:	74 2f                	je     402a7f <__sprintf_chk@plt+0x1ef>
  402a50:	b9 04 00 00 00       	mov    $0x4,%ecx
  402a55:	ba 60 64 41 00       	mov    $0x416460,%edx
  402a5a:	be 80 64 41 00       	mov    $0x416480,%esi
  402a5f:	48 89 c7             	mov    %rax,%rdi
  402a62:	e8 e9 73 00 00       	callq  409e50 <__sprintf_chk@plt+0x75c0>
  402a67:	85 c0                	test   %eax,%eax
  402a69:	0f 88 17 0d 00 00    	js     403786 <__sprintf_chk@plt+0xef6>
  402a6f:	48 98                	cltq   
  402a71:	31 ff                	xor    %edi,%edi
  402a73:	8b 34 85 60 64 41 00 	mov    0x416460(,%rax,4),%esi
  402a7a:	e8 c1 bb 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  402a7f:	bf 3c 38 41 00       	mov    $0x41383c,%edi
  402a84:	48 c7 05 39 86 21 00 	movq   $0x50,0x218639(%rip)        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  402a8b:	50 00 00 00 
  402a8f:	e8 2c f7 ff ff       	callq  4021c0 <getenv@plt>
  402a94:	49 89 c4             	mov    %rax,%r12
  402a97:	48 8d 44 24 40       	lea    0x40(%rsp),%rax
  402a9c:	4d 85 e4             	test   %r12,%r12
  402a9f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  402aa4:	74 0b                	je     402ab1 <__sprintf_chk@plt+0x221>
  402aa6:	41 80 3c 24 00       	cmpb   $0x0,(%r12)
  402aab:	0f 85 07 0d 00 00    	jne    4037b8 <__sprintf_chk@plt+0xf28>
  402ab1:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  402ab6:	31 c0                	xor    %eax,%eax
  402ab8:	be 13 54 00 00       	mov    $0x5413,%esi
  402abd:	bf 01 00 00 00       	mov    $0x1,%edi
  402ac2:	e8 e9 f9 ff ff       	callq  4024b0 <ioctl@plt>
  402ac7:	83 f8 ff             	cmp    $0xffffffff,%eax
  402aca:	74 11                	je     402add <__sprintf_chk@plt+0x24d>
  402acc:	0f b7 44 24 42       	movzwl 0x42(%rsp),%eax
  402ad1:	66 85 c0             	test   %ax,%ax
  402ad4:	74 07                	je     402add <__sprintf_chk@plt+0x24d>
  402ad6:	48 89 05 eb 85 21 00 	mov    %rax,0x2185eb(%rip)        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  402add:	bf 44 38 41 00       	mov    $0x413844,%edi
  402ae2:	e8 d9 f6 ff ff       	callq  4021c0 <getenv@plt>
  402ae7:	48 85 c0             	test   %rax,%rax
  402aea:	49 89 c4             	mov    %rax,%r12
  402aed:	48 c7 05 e0 85 21 00 	movq   $0x8,0x2185e0(%rip)        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  402af4:	08 00 00 00 
  402af8:	74 28                	je     402b22 <__sprintf_chk@plt+0x292>
  402afa:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  402aff:	45 31 c0             	xor    %r8d,%r8d
  402b02:	31 d2                	xor    %edx,%edx
  402b04:	31 f6                	xor    %esi,%esi
  402b06:	48 89 c7             	mov    %rax,%rdi
  402b09:	e8 82 e3 00 00       	callq  410e90 <__sprintf_chk@plt+0xe600>
  402b0e:	85 c0                	test   %eax,%eax
  402b10:	0f 85 a5 16 00 00    	jne    4041bb <__sprintf_chk@plt+0x192b>
  402b16:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  402b1b:	48 89 05 b6 85 21 00 	mov    %rax,0x2185b6(%rip)        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  402b22:	45 31 f6             	xor    %r14d,%r14d
  402b25:	45 31 ed             	xor    %r13d,%r13d
  402b28:	45 31 e4             	xor    %r12d,%r12d
  402b2b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  402b30:	4c 8d 44 24 38       	lea    0x38(%rsp),%r8
  402b35:	b9 80 30 41 00       	mov    $0x413080,%ecx
  402b3a:	ba c8 5b 41 00       	mov    $0x415bc8,%edx
  402b3f:	48 89 ee             	mov    %rbp,%rsi
  402b42:	89 df                	mov    %ebx,%edi
  402b44:	c7 44 24 38 ff ff ff 	movl   $0xffffffff,0x38(%rsp)
  402b4b:	ff 
  402b4c:	e8 5f f8 ff ff       	callq  4023b0 <getopt_long@plt>
  402b51:	83 f8 ff             	cmp    $0xffffffff,%eax
  402b54:	0f 84 c4 06 00 00    	je     40321e <__sprintf_chk@plt+0x98e>
  402b5a:	05 83 00 00 00       	add    $0x83,%eax
  402b5f:	3d 12 01 00 00       	cmp    $0x112,%eax
  402b64:	0f 87 8f 06 00 00    	ja     4031f9 <__sprintf_chk@plt+0x969>
  402b6a:	ff 24 c5 30 23 41 00 	jmpq   *0x412330(,%rax,8)
  402b71:	c6 05 cd 85 21 00 01 	movb   $0x1,0x2185cd(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  402b78:	c7 05 ce 85 21 00 00 	movl   $0x0,0x2185ce(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402b7f:	00 00 00 
  402b82:	eb ac                	jmp    402b30 <__sprintf_chk@plt+0x2a0>
  402b84:	41 be 01 00 00 00    	mov    $0x1,%r14d
  402b8a:	eb a4                	jmp    402b30 <__sprintf_chk@plt+0x2a0>
  402b8c:	c6 05 81 85 21 00 01 	movb   $0x1,0x218581(%rip)        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  402b93:	eb 9b                	jmp    402b30 <__sprintf_chk@plt+0x2a0>
  402b95:	c7 05 a1 85 21 00 b0 	movl   $0xb0,0x2185a1(%rip)        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  402b9c:	00 00 00 
  402b9f:	c7 05 8b 85 21 00 b0 	movl   $0xb0,0x21858b(%rip)        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  402ba6:	00 00 00 
  402ba9:	48 c7 05 84 85 21 00 	movq   $0x1,0x218584(%rip)        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  402bb0:	01 00 00 00 
  402bb4:	48 c7 05 a1 79 21 00 	movq   $0x1,0x2179a1(%rip)        # 61a560 <_fini@@Base+0x208664>
  402bbb:	01 00 00 00 
  402bbf:	e9 6c ff ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402bc4:	c7 05 82 85 21 00 00 	movl   $0x0,0x218582(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402bcb:	00 00 00 
  402bce:	c6 05 94 79 21 00 00 	movb   $0x0,0x217994(%rip)        # 61a569 <_fini@@Base+0x20866d>
  402bd5:	e9 56 ff ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402bda:	83 3d 6f 85 21 00 00 	cmpl   $0x0,0x21856f(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402be1:	c7 05 1d 85 21 00 02 	movl   $0x2,0x21851d(%rip)        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  402be8:	00 00 00 
  402beb:	c7 05 53 85 21 00 ff 	movl   $0xffffffff,0x218553(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  402bf2:	ff ff ff 
  402bf5:	0f 84 3b 10 00 00    	je     403c36 <__sprintf_chk@plt+0x13a6>
  402bfb:	c6 05 42 85 21 00 00 	movb   $0x0,0x218542(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  402c02:	c6 05 20 85 21 00 00 	movb   $0x0,0x218520(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  402c09:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  402c0f:	e9 1c ff ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c14:	c6 05 f2 84 21 00 01 	movb   $0x1,0x2184f2(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  402c1b:	e9 10 ff ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c20:	c7 05 22 85 21 00 01 	movl   $0x1,0x218522(%rip)        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  402c27:	00 00 00 
  402c2a:	e9 01 ff ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c2f:	be 05 00 00 00       	mov    $0x5,%esi
  402c34:	31 ff                	xor    %edi,%edi
  402c36:	e8 05 ba 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  402c3b:	e9 f0 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c40:	c7 05 be 84 21 00 02 	movl   $0x2,0x2184be(%rip)        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  402c47:	00 00 00 
  402c4a:	e9 e1 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c4f:	c6 05 27 85 21 00 01 	movb   $0x1,0x218527(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  402c56:	e9 d5 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c5b:	c7 05 e3 84 21 00 01 	movl   $0x1,0x2184e3(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  402c62:	00 00 00 
  402c65:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  402c6b:	e9 c0 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c70:	c7 05 ce 84 21 00 ff 	movl   $0xffffffff,0x2184ce(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  402c77:	ff ff ff 
  402c7a:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  402c80:	e9 ab fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402c85:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  402c8a:	48 8b 3d af 79 21 00 	mov    0x2179af(%rip),%rdi        # 61a640 <optarg@@GLIBC_2.2.5>
  402c91:	45 31 c0             	xor    %r8d,%r8d
  402c94:	31 d2                	xor    %edx,%edx
  402c96:	31 f6                	xor    %esi,%esi
  402c98:	e8 f3 e1 00 00       	callq  410e90 <__sprintf_chk@plt+0xe600>
  402c9d:	85 c0                	test   %eax,%eax
  402c9f:	0f 85 58 0f 00 00    	jne    403bfd <__sprintf_chk@plt+0x136d>
  402ca5:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  402caa:	48 89 05 27 84 21 00 	mov    %rax,0x218427(%rip)        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  402cb1:	e9 7a fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402cb6:	c7 05 88 84 21 00 02 	movl   $0x2,0x218488(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  402cbd:	00 00 00 
  402cc0:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  402cc6:	e9 65 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402ccb:	c6 05 3c 84 21 00 01 	movb   $0x1,0x21843c(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  402cd2:	e9 59 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402cd7:	be 03 00 00 00       	mov    $0x3,%esi
  402cdc:	31 ff                	xor    %edi,%edi
  402cde:	e8 5d b9 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  402ce3:	e9 48 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402ce8:	31 f6                	xor    %esi,%esi
  402cea:	31 ff                	xor    %edi,%edi
  402cec:	e8 4f b9 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  402cf1:	e9 3a fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402cf6:	c7 05 10 84 21 00 05 	movl   $0x5,0x218410(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  402cfd:	00 00 00 
  402d00:	e9 2b fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d05:	bf 10 00 00 00       	mov    $0x10,%edi
  402d0a:	4c 8b 3d 2f 79 21 00 	mov    0x21792f(%rip),%r15        # 61a640 <optarg@@GLIBC_2.2.5>
  402d11:	e8 2a df 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  402d16:	48 8b 15 e3 83 21 00 	mov    0x2183e3(%rip),%rdx        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402d1d:	4c 89 38             	mov    %r15,(%rax)
  402d20:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402d24:	48 89 05 d5 83 21 00 	mov    %rax,0x2183d5(%rip)        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402d2b:	e9 00 fe ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d30:	c7 05 d6 83 21 00 03 	movl   $0x3,0x2183d6(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  402d37:	00 00 00 
  402d3a:	e9 f1 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d3f:	c6 05 22 78 21 00 00 	movb   $0x0,0x217822(%rip)        # 61a568 <_fini@@Base+0x20866c>
  402d46:	e9 e5 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d4b:	c7 05 d7 83 21 00 03 	movl   $0x3,0x2183d7(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  402d52:	00 00 00 
  402d55:	e9 d6 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d5a:	c6 05 cf 83 21 00 01 	movb   $0x1,0x2183cf(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  402d61:	e9 ca fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d66:	c7 05 e0 83 21 00 02 	movl   $0x2,0x2183e0(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402d6d:	00 00 00 
  402d70:	e9 bb fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402d75:	bf 10 00 00 00       	mov    $0x10,%edi
  402d7a:	e8 c1 de 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  402d7f:	48 8b 15 7a 83 21 00 	mov    0x21837a(%rip),%rdx        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402d86:	48 c7 00 64 38 41 00 	movq   $0x413864,(%rax)
  402d8d:	bf 10 00 00 00       	mov    $0x10,%edi
  402d92:	48 89 05 67 83 21 00 	mov    %rax,0x218367(%rip)        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402d99:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402d9d:	e8 9e de 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  402da2:	48 8b 15 57 83 21 00 	mov    0x218357(%rip),%rdx        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402da9:	48 c7 00 63 38 41 00 	movq   $0x413863,(%rax)
  402db0:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402db4:	48 89 05 45 83 21 00 	mov    %rax,0x218345(%rip)        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  402dbb:	e9 70 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402dc0:	83 3d 41 83 21 00 00 	cmpl   $0x0,0x218341(%rip)        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  402dc7:	0f 85 63 fd ff ff    	jne    402b30 <__sprintf_chk@plt+0x2a0>
  402dcd:	c7 05 31 83 21 00 01 	movl   $0x1,0x218331(%rip)        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  402dd4:	00 00 00 
  402dd7:	e9 54 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402ddc:	83 3d 6d 83 21 00 00 	cmpl   $0x0,0x21836d(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402de3:	0f 84 47 fd ff ff    	je     402b30 <__sprintf_chk@plt+0x2a0>
  402de9:	c7 05 5d 83 21 00 01 	movl   $0x1,0x21835d(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402df0:	00 00 00 
  402df3:	e9 38 fd ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402df8:	31 ff                	xor    %edi,%edi
  402dfa:	e8 51 69 00 00       	callq  409750 <__sprintf_chk@plt+0x6ec0>
  402dff:	8b 05 67 77 21 00    	mov    0x217767(%rip),%eax        # 61a56c <_fini@@Base+0x208670>
  402e05:	48 8b 0d 64 77 21 00 	mov    0x217764(%rip),%rcx        # 61a570 <_fini@@Base+0x208674>
  402e0c:	83 f8 01             	cmp    $0x1,%eax
  402e0f:	0f 84 de 0d 00 00    	je     403bf3 <__sprintf_chk@plt+0x1363>
  402e15:	83 f8 02             	cmp    $0x2,%eax
  402e18:	be 0f 38 41 00       	mov    $0x41380f,%esi
  402e1d:	b8 0e 38 41 00       	mov    $0x41380e,%eax
  402e22:	48 0f 45 f0          	cmovne %rax,%rsi
  402e26:	48 8b 3d e3 77 21 00 	mov    0x2177e3(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  402e2d:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  402e34:	00 
  402e35:	41 b9 bd 38 41 00    	mov    $0x4138bd,%r9d
  402e3b:	41 b8 cd 38 41 00    	mov    $0x4138cd,%r8d
  402e41:	ba fc 37 41 00       	mov    $0x4137fc,%edx
  402e46:	31 c0                	xor    %eax,%eax
  402e48:	e8 e3 dc 00 00       	callq  410b30 <__sprintf_chk@plt+0xe2a0>
  402e4d:	31 ff                	xor    %edi,%edi
  402e4f:	e8 9c f9 ff ff       	callq  4027f0 <exit@plt>
  402e54:	4c 8b 25 e5 77 21 00 	mov    0x2177e5(%rip),%r12        # 61a640 <optarg@@GLIBC_2.2.5>
  402e5b:	e9 d0 fc ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402e60:	4c 8b 0d 11 77 21 00 	mov    0x217711(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  402e67:	48 8b 35 d2 77 21 00 	mov    0x2177d2(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  402e6e:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  402e74:	b9 50 2f 41 00       	mov    $0x412f50,%ecx
  402e79:	ba 80 2f 41 00       	mov    $0x412f80,%edx
  402e7e:	bf 83 38 41 00       	mov    $0x413883,%edi
  402e83:	e8 98 72 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  402e88:	8b 04 85 50 2f 41 00 	mov    0x412f50(,%rax,4),%eax
  402e8f:	89 05 b7 82 21 00    	mov    %eax,0x2182b7(%rip)        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  402e95:	e9 96 fc ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402e9a:	4c 8b 0d d7 76 21 00 	mov    0x2176d7(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  402ea1:	48 8b 35 98 77 21 00 	mov    0x217798(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  402ea8:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  402eae:	b9 b0 2f 41 00       	mov    $0x412fb0,%ecx
  402eb3:	ba e0 2f 41 00       	mov    $0x412fe0,%edx
  402eb8:	bf 7c 38 41 00       	mov    $0x41387c,%edi
  402ebd:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  402ec3:	e8 58 72 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  402ec8:	8b 04 85 b0 2f 41 00 	mov    0x412fb0(,%rax,4),%eax
  402ecf:	89 05 73 82 21 00    	mov    %eax,0x218273(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  402ed5:	e9 56 fc ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402eda:	c7 05 5c 82 21 00 90 	movl   $0x90,0x21825c(%rip)        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  402ee1:	00 00 00 
  402ee4:	c7 05 46 82 21 00 90 	movl   $0x90,0x218246(%rip)        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  402eeb:	00 00 00 
  402eee:	48 c7 05 3f 82 21 00 	movq   $0x1,0x21823f(%rip)        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  402ef5:	01 00 00 00 
  402ef9:	48 c7 05 5c 76 21 00 	movq   $0x1,0x21765c(%rip)        # 61a560 <_fini@@Base+0x208664>
  402f00:	01 00 00 00 
  402f04:	e9 27 fc ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402f09:	c6 05 e0 81 21 00 00 	movb   $0x0,0x2181e0(%rip)        # 61b0f0 <stderr@@GLIBC_2.2.5+0xaa0>
  402f10:	e9 1b fc ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402f15:	4c 8b 0d 5c 76 21 00 	mov    0x21765c(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  402f1c:	48 8b 35 1d 77 21 00 	mov    0x21771d(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  402f23:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  402f29:	b9 60 64 41 00       	mov    $0x416460,%ecx
  402f2e:	ba 80 64 41 00       	mov    $0x416480,%edx
  402f33:	bf ad 38 41 00       	mov    $0x4138ad,%edi
  402f38:	e8 e3 71 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  402f3d:	8b 34 85 60 64 41 00 	mov    0x416460(,%rax,4),%esi
  402f44:	31 ff                	xor    %edi,%edi
  402f46:	e8 f5 b6 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  402f4b:	e9 e0 fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402f50:	4c 8b 0d 21 76 21 00 	mov    0x217621(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  402f57:	48 8b 35 e2 76 21 00 	mov    0x2176e2(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  402f5e:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  402f64:	b9 b0 36 41 00       	mov    $0x4136b0,%ecx
  402f69:	ba c0 36 41 00       	mov    $0x4136c0,%edx
  402f6e:	bf 9b 38 41 00       	mov    $0x41389b,%edi
  402f73:	e8 a8 71 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  402f78:	8b 04 85 b0 36 41 00 	mov    0x4136b0(,%rax,4),%eax
  402f7f:	89 05 a7 81 21 00    	mov    %eax,0x2181a7(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  402f85:	e9 a6 fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402f8a:	bf 10 00 00 00       	mov    $0x10,%edi
  402f8f:	e8 ac dc 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  402f94:	48 8b 15 a5 76 21 00 	mov    0x2176a5(%rip),%rdx        # 61a640 <optarg@@GLIBC_2.2.5>
  402f9b:	48 89 10             	mov    %rdx,(%rax)
  402f9e:	48 8b 15 53 81 21 00 	mov    0x218153(%rip),%rdx        # 61b0f8 <stderr@@GLIBC_2.2.5+0xaa8>
  402fa5:	48 89 05 4c 81 21 00 	mov    %rax,0x21814c(%rip)        # 61b0f8 <stderr@@GLIBC_2.2.5+0xaa8>
  402fac:	48 89 50 08          	mov    %rdx,0x8(%rax)
  402fb0:	e9 7b fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402fb5:	c6 05 50 81 21 00 01 	movb   $0x1,0x218150(%rip)        # 61b10c <stderr@@GLIBC_2.2.5+0xabc>
  402fbc:	e9 6f fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402fc1:	c7 05 85 81 21 00 00 	movl   $0x0,0x218185(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  402fc8:	00 00 00 
  402fcb:	41 bc 13 38 41 00    	mov    $0x413813,%r12d
  402fd1:	e9 5a fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  402fd6:	4c 8b 0d 9b 75 21 00 	mov    0x21759b(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  402fdd:	48 8b 35 5c 76 21 00 	mov    0x21765c(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  402fe4:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  402fea:	b9 10 30 41 00       	mov    $0x413010,%ecx
  402fef:	ba 40 30 41 00       	mov    $0x413040,%edx
  402ff4:	bf 8a 38 41 00       	mov    $0x41388a,%edi
  402ff9:	e8 22 71 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  402ffe:	8b 04 85 10 30 41 00 	mov    0x413010(,%rax,4),%eax
  403005:	89 05 45 81 21 00    	mov    %eax,0x218145(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  40300b:	e9 20 fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403010:	c7 05 12 81 21 00 02 	movl   $0x2,0x218112(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  403017:	00 00 00 
  40301a:	e9 11 fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  40301f:	c7 05 e7 80 21 00 04 	movl   $0x4,0x2180e7(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  403026:	00 00 00 
  403029:	e9 02 fb ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  40302e:	48 8b 35 0b 76 21 00 	mov    0x21760b(%rip),%rsi        # 61a640 <optarg@@GLIBC_2.2.5>
  403035:	48 85 f6             	test   %rsi,%rsi
  403038:	0f 84 3e 0a 00 00    	je     403a7c <__sprintf_chk@plt+0x11ec>
  40303e:	4c 8b 0d 33 75 21 00 	mov    0x217533(%rip),%r9        # 61a578 <_fini@@Base+0x20867c>
  403045:	41 b8 04 00 00 00    	mov    $0x4,%r8d
  40304b:	b9 c0 2e 41 00       	mov    $0x412ec0,%ecx
  403050:	ba 00 2f 41 00       	mov    $0x412f00,%edx
  403055:	bf 93 38 41 00       	mov    $0x413893,%edi
  40305a:	e8 c1 70 00 00       	callq  40a120 <__sprintf_chk@plt+0x7890>
  40305f:	8b 04 85 c0 2e 41 00 	mov    0x412ec0(,%rax,4),%eax
  403066:	83 f8 01             	cmp    $0x1,%eax
  403069:	0f 84 0d 0a 00 00    	je     403a7c <__sprintf_chk@plt+0x11ec>
  40306f:	83 f8 02             	cmp    $0x2,%eax
  403072:	0f 84 f2 09 00 00    	je     403a6a <__sprintf_chk@plt+0x11da>
  403078:	c6 05 aa 80 21 00 00 	movb   $0x0,0x2180aa(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  40307f:	e9 ac fa ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403084:	48 8b 3d b5 75 21 00 	mov    0x2175b5(%rip),%rdi        # 61a640 <optarg@@GLIBC_2.2.5>
  40308b:	ba 38 b1 61 00       	mov    $0x61b138,%edx
  403090:	be 40 b1 61 00       	mov    $0x61b140,%esi
  403095:	e8 76 97 00 00       	callq  40c810 <__sprintf_chk@plt+0x9f80>
  40309a:	85 c0                	test   %eax,%eax
  40309c:	0f 85 1e 13 00 00    	jne    4043c0 <__sprintf_chk@plt+0x1b30>
  4030a2:	8b 05 98 80 21 00    	mov    0x218098(%rip),%eax        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  4030a8:	89 05 86 80 21 00    	mov    %eax,0x218086(%rip)        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  4030ae:	48 8b 05 83 80 21 00 	mov    0x218083(%rip),%rax        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  4030b5:	48 89 05 a4 74 21 00 	mov    %rax,0x2174a4(%rip)        # 61a560 <_fini@@Base+0x208664>
  4030bc:	e9 6f fa ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4030c1:	c6 05 7e 80 21 00 01 	movb   $0x1,0x21807e(%rip)        # 61b146 <stderr@@GLIBC_2.2.5+0xaf6>
  4030c8:	e9 63 fa ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4030cd:	c7 05 79 80 21 00 03 	movl   $0x3,0x218079(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4030d4:	00 00 00 
  4030d7:	e9 54 fa ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4030dc:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  4030e1:	48 8b 3d 58 75 21 00 	mov    0x217558(%rip),%rdi        # 61a640 <optarg@@GLIBC_2.2.5>
  4030e8:	45 31 c0             	xor    %r8d,%r8d
  4030eb:	31 d2                	xor    %edx,%edx
  4030ed:	31 f6                	xor    %esi,%esi
  4030ef:	e8 9c dd 00 00       	callq  410e90 <__sprintf_chk@plt+0xe600>
  4030f4:	85 c0                	test   %eax,%eax
  4030f6:	75 08                	jne    403100 <__sprintf_chk@plt+0x870>
  4030f8:	48 83 7c 24 40 00    	cmpq   $0x0,0x40(%rsp)
  4030fe:	75 34                	jne    403134 <__sprintf_chk@plt+0x8a4>
  403100:	48 8b 3d 39 75 21 00 	mov    0x217539(%rip),%rdi        # 61a640 <optarg@@GLIBC_2.2.5>
  403107:	e8 24 b8 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  40310c:	ba 05 00 00 00       	mov    $0x5,%edx
  403111:	49 89 c7             	mov    %rax,%r15
  403114:	be 4c 38 41 00       	mov    $0x41384c,%esi
  403119:	31 ff                	xor    %edi,%edi
  40311b:	e8 40 f2 ff ff       	callq  402360 <dcgettext@plt>
  403120:	4c 89 f9             	mov    %r15,%rcx
  403123:	48 89 c2             	mov    %rax,%rdx
  403126:	31 f6                	xor    %esi,%esi
  403128:	bf 02 00 00 00       	mov    $0x2,%edi
  40312d:	31 c0                	xor    %eax,%eax
  40312f:	e8 3c f6 ff ff       	callq  402770 <error@plt>
  403134:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  403139:	48 89 05 88 7f 21 00 	mov    %rax,0x217f88(%rip)        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  403140:	e9 eb f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403145:	c7 05 f9 7f 21 00 03 	movl   $0x3,0x217ff9(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  40314c:	00 00 00 
  40314f:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  403155:	e9 d6 f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  40315a:	c7 05 e8 7f 21 00 02 	movl   $0x2,0x217fe8(%rip)        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  403161:	00 00 00 
  403164:	e9 c7 f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403169:	c7 05 d5 7f 21 00 04 	movl   $0x4,0x217fd5(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  403170:	00 00 00 
  403173:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  403179:	e9 b2 f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  40317e:	c6 05 bf 7f 21 00 01 	movb   $0x1,0x217fbf(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  403185:	e9 a6 f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  40318a:	c6 05 b6 7f 21 00 01 	movb   $0x1,0x217fb6(%rip)        # 61b147 <stderr@@GLIBC_2.2.5+0xaf7>
  403191:	e9 9a f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403196:	c6 05 53 7f 21 00 01 	movb   $0x1,0x217f53(%rip)        # 61b0f0 <stderr@@GLIBC_2.2.5+0xaa0>
  40319d:	e9 8e f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4031a2:	c7 05 80 7f 21 00 01 	movl   $0x1,0x217f80(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  4031a9:	00 00 00 
  4031ac:	e9 7f f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4031b1:	c7 05 95 7f 21 00 00 	movl   $0x0,0x217f95(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4031b8:	00 00 00 
  4031bb:	c6 05 a6 73 21 00 00 	movb   $0x0,0x2173a6(%rip)        # 61a568 <_fini@@Base+0x20866c>
  4031c2:	e9 69 f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4031c7:	c7 05 7f 7f 21 00 04 	movl   $0x4,0x217f7f(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4031ce:	00 00 00 
  4031d1:	e9 5a f9 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  4031d6:	48 8b 1d 73 74 21 00 	mov    0x217473(%rip),%rbx        # 61a650 <stderr@@GLIBC_2.2.5>
  4031dd:	be f8 5b 41 00       	mov    $0x415bf8,%esi
  4031e2:	31 ff                	xor    %edi,%edi
  4031e4:	ba 05 00 00 00       	mov    $0x5,%edx
  4031e9:	e8 72 f1 ff ff       	callq  402360 <dcgettext@plt>
  4031ee:	48 89 de             	mov    %rbx,%rsi
  4031f1:	48 89 c7             	mov    %rax,%rdi
  4031f4:	e8 27 f3 ff ff       	callq  402520 <fputs_unlocked@plt>
  4031f9:	bf 02 00 00 00       	mov    $0x2,%edi
  4031fe:	e8 4d 65 00 00       	callq  409750 <__sprintf_chk@plt+0x6ec0>
  403203:	be 05 00 00 00       	mov    $0x5,%esi
  403208:	31 ff                	xor    %edi,%edi
  40320a:	c7 05 3c 7f 21 00 02 	movl   $0x2,0x217f3c(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403211:	00 00 00 
  403214:	e8 27 b4 00 00       	callq  40e640 <__sprintf_chk@plt+0xbdb0>
  403219:	e9 a7 f7 ff ff       	jmpq   4029c5 <__sprintf_chk@plt+0x135>
  40321e:	48 83 3d 12 7f 21 00 	cmpq   $0x0,0x217f12(%rip)        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  403225:	00 
  403226:	0f 84 5e 07 00 00    	je     40398a <__sprintf_chk@plt+0x10fa>
  40322c:	48 8b 15 95 7e 21 00 	mov    0x217e95(%rip),%rdx        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  403233:	b8 01 00 00 00       	mov    $0x1,%eax
  403238:	48 83 fa 02          	cmp    $0x2,%rdx
  40323c:	0f 87 b9 05 00 00    	ja     4037fb <__sprintf_chk@plt+0xf6b>
  403242:	31 ff                	xor    %edi,%edi
  403244:	48 89 05 d5 7d 21 00 	mov    %rax,0x217dd5(%rip)        # 61b020 <stderr@@GLIBC_2.2.5+0x9d0>
  40324b:	e8 b0 b3 00 00       	callq  40e600 <__sprintf_chk@plt+0xbd70>
  403250:	48 89 c7             	mov    %rax,%rdi
  403253:	48 89 05 8e 7e 21 00 	mov    %rax,0x217e8e(%rip)        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  40325a:	e8 d1 b3 00 00       	callq  40e630 <__sprintf_chk@plt+0xbda0>
  40325f:	83 f8 05             	cmp    $0x5,%eax
  403262:	0f 84 04 10 00 00    	je     40426c <__sprintf_chk@plt+0x19dc>
  403268:	8b 05 be 7e 21 00    	mov    0x217ebe(%rip),%eax        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  40326e:	83 f8 01             	cmp    $0x1,%eax
  403271:	76 36                	jbe    4032a9 <__sprintf_chk@plt+0xa19>
  403273:	4c 8d b0 ed 38 41 00 	lea    0x4138ed(%rax),%r14
  40327a:	48 83 e8 02          	sub    $0x2,%rax
  40327e:	0f b6 80 ef 38 41 00 	movzbl 0x4138ef(%rax),%eax
  403285:	84 c0                	test   %al,%al
  403287:	74 20                	je     4032a9 <__sprintf_chk@plt+0xa19>
  403289:	48 8b 3d 58 7e 21 00 	mov    0x217e58(%rip),%rdi        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  403290:	49 83 c6 01          	add    $0x1,%r14
  403294:	0f be f0             	movsbl %al,%esi
  403297:	ba 01 00 00 00       	mov    $0x1,%edx
  40329c:	e8 af b3 00 00       	callq  40e650 <__sprintf_chk@plt+0xbdc0>
  4032a1:	41 0f b6 06          	movzbl (%r14),%eax
  4032a5:	84 c0                	test   %al,%al
  4032a7:	75 e0                	jne    403289 <__sprintf_chk@plt+0x9f9>
  4032a9:	31 ff                	xor    %edi,%edi
  4032ab:	e8 50 b3 00 00       	callq  40e600 <__sprintf_chk@plt+0xbd70>
  4032b0:	ba 01 00 00 00       	mov    $0x1,%edx
  4032b5:	be 3a 00 00 00       	mov    $0x3a,%esi
  4032ba:	48 89 c7             	mov    %rax,%rdi
  4032bd:	48 89 05 1c 7e 21 00 	mov    %rax,0x217e1c(%rip)        # 61b0e0 <stderr@@GLIBC_2.2.5+0xa90>
  4032c4:	e8 87 b3 00 00       	callq  40e650 <__sprintf_chk@plt+0xbdc0>
  4032c9:	80 3d 60 7e 21 00 00 	cmpb   $0x0,0x217e60(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  4032d0:	74 10                	je     4032e2 <__sprintf_chk@plt+0xa52>
  4032d2:	83 3d 77 7e 21 00 00 	cmpl   $0x0,0x217e77(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4032d9:	74 07                	je     4032e2 <__sprintf_chk@plt+0xa52>
  4032db:	c6 05 4e 7e 21 00 00 	movb   $0x0,0x217e4e(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  4032e2:	8b 05 64 7e 21 00    	mov    0x217e64(%rip),%eax        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  4032e8:	83 e8 01             	sub    $0x1,%eax
  4032eb:	83 f8 01             	cmp    $0x1,%eax
  4032ee:	0f 86 71 06 00 00    	jbe    403965 <__sprintf_chk@plt+0x10d5>
  4032f4:	83 3d 55 7e 21 00 00 	cmpl   $0x0,0x217e55(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4032fb:	0f 84 0c 05 00 00    	je     40380d <__sprintf_chk@plt+0xf7d>
  403301:	80 3d 21 7e 21 00 00 	cmpb   $0x0,0x217e21(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  403308:	44 8b 25 11 73 21 00 	mov    0x217311(%rip),%r12d        # 61a620 <optind@@GLIBC_2.2.5>
  40330f:	0f 85 b0 07 00 00    	jne    403ac5 <__sprintf_chk@plt+0x1235>
  403315:	83 3d f4 7d 21 00 01 	cmpl   $0x1,0x217df4(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  40331c:	0f 84 17 07 00 00    	je     403a39 <__sprintf_chk@plt+0x11a9>
  403322:	80 3d e5 7d 21 00 00 	cmpb   $0x0,0x217de5(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  403329:	0f 85 c0 06 00 00    	jne    4039ef <__sprintf_chk@plt+0x115f>
  40332f:	8b 05 13 7e 21 00    	mov    0x217e13(%rip),%eax        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  403335:	83 f8 04             	cmp    $0x4,%eax
  403338:	0f 84 3c 04 00 00    	je     40377a <__sprintf_chk@plt+0xeea>
  40333e:	83 f8 02             	cmp    $0x2,%eax
  403341:	0f 84 33 04 00 00    	je     40377a <__sprintf_chk@plt+0xeea>
  403347:	83 3d 02 7e 21 00 00 	cmpl   $0x0,0x217e02(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  40334e:	0f 84 26 04 00 00    	je     40377a <__sprintf_chk@plt+0xeea>
  403354:	80 3d 22 7e 21 00 00 	cmpb   $0x0,0x217e22(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  40335b:	0f 85 19 04 00 00    	jne    40377a <__sprintf_chk@plt+0xeea>
  403361:	80 3d dc 7d 21 00 00 	cmpb   $0x0,0x217ddc(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  403368:	0f 85 0c 04 00 00    	jne    40377a <__sprintf_chk@plt+0xeea>
  40336e:	80 3d 99 7d 21 00 00 	cmpb   $0x0,0x217d99(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  403375:	c6 05 45 7d 21 00 00 	movb   $0x0,0x217d45(%rip)        # 61b0c1 <stderr@@GLIBC_2.2.5+0xa71>
  40337c:	b8 01 00 00 00       	mov    $0x1,%eax
  403381:	75 1d                	jne    4033a0 <__sprintf_chk@plt+0xb10>
  403383:	80 3d 9f 7d 21 00 00 	cmpb   $0x0,0x217d9f(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  40338a:	75 14                	jne    4033a0 <__sprintf_chk@plt+0xb10>
  40338c:	83 3d 99 7d 21 00 00 	cmpl   $0x0,0x217d99(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  403393:	75 0b                	jne    4033a0 <__sprintf_chk@plt+0xb10>
  403395:	80 3d 70 7d 21 00 00 	cmpb   $0x0,0x217d70(%rip)        # 61b10c <stderr@@GLIBC_2.2.5+0xabc>
  40339c:	75 02                	jne    4033a0 <__sprintf_chk@plt+0xb10>
  40339e:	31 c0                	xor    %eax,%eax
  4033a0:	88 05 1a 7d 21 00    	mov    %al,0x217d1a(%rip)        # 61b0c0 <stderr@@GLIBC_2.2.5+0xa70>
  4033a6:	80 25 13 7d 21 00 01 	andb   $0x1,0x217d13(%rip)        # 61b0c0 <stderr@@GLIBC_2.2.5+0xa70>
  4033ad:	80 3d 7c 7d 21 00 00 	cmpb   $0x0,0x217d7c(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  4033b4:	74 32                	je     4033e8 <__sprintf_chk@plt+0xb58>
  4033b6:	41 b8 f0 21 40 00    	mov    $0x4021f0,%r8d
  4033bc:	b9 40 26 40 00       	mov    $0x402640,%ecx
  4033c1:	31 d2                	xor    %edx,%edx
  4033c3:	31 f6                	xor    %esi,%esi
  4033c5:	bf c0 af 61 00       	mov    $0x61afc0,%edi
  4033ca:	e8 21 f0 ff ff       	callq  4023f0 <_obstack_begin@plt>
  4033cf:	41 b8 f0 21 40 00    	mov    $0x4021f0,%r8d
  4033d5:	b9 40 26 40 00       	mov    $0x402640,%ecx
  4033da:	31 d2                	xor    %edx,%edx
  4033dc:	31 f6                	xor    %esi,%esi
  4033de:	bf 60 af 61 00       	mov    $0x61af60,%edi
  4033e3:	e8 08 f0 ff ff       	callq  4023f0 <_obstack_begin@plt>
  4033e8:	41 89 dd             	mov    %ebx,%r13d
  4033eb:	bf 00 4b 00 00       	mov    $0x4b00,%edi
  4033f0:	48 c7 05 bd 7d 21 00 	movq   $0x64,0x217dbd(%rip)        # 61b1b8 <stderr@@GLIBC_2.2.5+0xb68>
  4033f7:	64 00 00 00 
  4033fb:	45 29 e5             	sub    %r12d,%r13d
  4033fe:	e8 3d d8 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  403403:	48 c7 05 a2 7d 21 00 	movq   $0x0,0x217da2(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  40340a:	00 00 00 00 
  40340e:	48 89 05 ab 7d 21 00 	mov    %rax,0x217dab(%rip)        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  403415:	e8 b6 19 00 00       	callq  404dd0 <__sprintf_chk@plt+0x2540>
  40341a:	45 85 ed             	test   %r13d,%r13d
  40341d:	0f 8e 7d 0e 00 00    	jle    4042a0 <__sprintf_chk@plt+0x1a10>
  403423:	49 63 c4             	movslq %r12d,%rax
  403426:	48 8d 6c c5 00       	lea    0x0(%rbp,%rax,8),%rbp
  40342b:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  40342f:	31 f6                	xor    %esi,%esi
  403431:	41 83 c4 01          	add    $0x1,%r12d
  403435:	b9 19 69 41 00       	mov    $0x416919,%ecx
  40343a:	ba 01 00 00 00       	mov    $0x1,%edx
  40343f:	48 83 c5 08          	add    $0x8,%rbp
  403443:	e8 58 4a 00 00       	callq  407ea0 <__sprintf_chk@plt+0x5610>
  403448:	44 39 e3             	cmp    %r12d,%ebx
  40344b:	7f de                	jg     40342b <__sprintf_chk@plt+0xb9b>
  40344d:	48 83 3d 5b 7d 21 00 	cmpq   $0x0,0x217d5b(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  403454:	00 
  403455:	0f 85 a5 0d 00 00    	jne    404200 <__sprintf_chk@plt+0x1970>
  40345b:	48 8b 05 2e 7d 21 00 	mov    0x217d2e(%rip),%rax        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  403462:	41 83 ed 01          	sub    $0x1,%r13d
  403466:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40346b:	7f 65                	jg     4034d2 <__sprintf_chk@plt+0xc42>
  40346d:	e9 13 0f 00 00       	jmpq   404385 <__sprintf_chk@plt+0x1af5>
  403472:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403478:	ba 05 00 00 00       	mov    $0x5,%edx
  40347d:	be e8 5c 41 00       	mov    $0x415ce8,%esi
  403482:	31 ff                	xor    %edi,%edi
  403484:	e8 d7 ee ff ff       	callq  402360 <dcgettext@plt>
  403489:	0f b6 7c 24 2f       	movzbl 0x2f(%rsp),%edi
  40348e:	4c 89 f2             	mov    %r14,%rdx
  403491:	48 89 c6             	mov    %rax,%rsi
  403494:	e8 77 23 00 00       	callq  405810 <__sprintf_chk@plt+0x2f80>
  403499:	4c 89 ef             	mov    %r13,%rdi
  40349c:	e8 3f f0 ff ff       	callq  4024e0 <closedir@plt>
  4034a1:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  4034a6:	48 8b 3b             	mov    (%rbx),%rdi
  4034a9:	e8 42 ed ff ff       	callq  4021f0 <free@plt>
  4034ae:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  4034b2:	e8 39 ed ff ff       	callq  4021f0 <free@plt>
  4034b7:	48 89 df             	mov    %rbx,%rdi
  4034ba:	e8 31 ed ff ff       	callq  4021f0 <free@plt>
  4034bf:	c6 05 0a 7c 21 00 01 	movb   $0x1,0x217c0a(%rip)        # 61b0d0 <stderr@@GLIBC_2.2.5+0xa80>
  4034c6:	48 8b 05 c3 7c 21 00 	mov    0x217cc3(%rip),%rax        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  4034cd:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4034d2:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  4034d8:	0f 84 d7 0b 00 00    	je     4040b5 <__sprintf_chk@plt+0x1825>
  4034de:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  4034e3:	48 83 3d dd 7c 21 00 	cmpq   $0x0,0x217cdd(%rip)        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  4034ea:	00 
  4034eb:	48 8b 41 18          	mov    0x18(%rcx),%rax
  4034ef:	48 89 05 9a 7c 21 00 	mov    %rax,0x217c9a(%rip)        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  4034f6:	0f 84 b9 0a 00 00    	je     403fb5 <__sprintf_chk@plt+0x1725>
  4034fc:	4c 8b 31             	mov    (%rcx),%r14
  4034ff:	4d 85 f6             	test   %r14,%r14
  403502:	0f 84 ba 0a 00 00    	je     403fc2 <__sprintf_chk@plt+0x1732>
  403508:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40350d:	0f b6 48 10          	movzbl 0x10(%rax),%ecx
  403511:	48 8b 58 08          	mov    0x8(%rax),%rbx
  403515:	88 4c 24 2f          	mov    %cl,0x2f(%rsp)
  403519:	e8 12 ed ff ff       	callq  402230 <__errno_location@plt>
  40351e:	4c 89 f7             	mov    %r14,%rdi
  403521:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  403527:	49 89 c4             	mov    %rax,%r12
  40352a:	e8 f1 ed ff ff       	callq  402320 <opendir@plt>
  40352f:	48 85 c0             	test   %rax,%rax
  403532:	49 89 c5             	mov    %rax,%r13
  403535:	0f 84 5a 0c 00 00    	je     404195 <__sprintf_chk@plt+0x1905>
  40353b:	48 83 3d 85 7c 21 00 	cmpq   $0x0,0x217c85(%rip)        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  403542:	00 
  403543:	0f 84 b6 00 00 00    	je     4035ff <__sprintf_chk@plt+0xd6f>
  403549:	48 89 c7             	mov    %rax,%rdi
  40354c:	e8 1f f0 ff ff       	callq  402570 <dirfd@plt>
  403551:	85 c0                	test   %eax,%eax
  403553:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  403558:	0f 88 37 09 00 00    	js     403e95 <__sprintf_chk@plt+0x1605>
  40355e:	89 c6                	mov    %eax,%esi
  403560:	bf 01 00 00 00       	mov    $0x1,%edi
  403565:	e8 16 f1 ff ff       	callq  402680 <__fxstat@plt>
  40356a:	c1 e8 1f             	shr    $0x1f,%eax
  40356d:	84 c0                	test   %al,%al
  40356f:	0f 85 03 ff ff ff    	jne    403478 <__sprintf_chk@plt+0xbe8>
  403575:	48 8b 4c 24 48       	mov    0x48(%rsp),%rcx
  40357a:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
  40357f:	bf 10 00 00 00       	mov    $0x10,%edi
  403584:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  403589:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
  40358e:	e8 ad d6 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  403593:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
  403598:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  40359d:	48 89 c6             	mov    %rax,%rsi
  4035a0:	48 8b 3d 21 7c 21 00 	mov    0x217c21(%rip),%rdi        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  4035a7:	48 89 c5             	mov    %rax,%rbp
  4035aa:	48 89 08             	mov    %rcx,(%rax)
  4035ad:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4035b1:	e8 9a 85 00 00       	callq  40bb50 <__sprintf_chk@plt+0x92c0>
  4035b6:	48 85 c0             	test   %rax,%rax
  4035b9:	0f 84 fc 0d 00 00    	je     4043bb <__sprintf_chk@plt+0x1b2b>
  4035bf:	48 39 c5             	cmp    %rax,%rbp
  4035c2:	0f 85 1a 09 00 00    	jne    403ee2 <__sprintf_chk@plt+0x1652>
  4035c8:	48 8b 05 49 79 21 00 	mov    0x217949(%rip),%rax        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  4035cf:	48 8b 15 4a 79 21 00 	mov    0x21794a(%rip),%rdx        # 61af20 <stderr@@GLIBC_2.2.5+0x8d0>
  4035d6:	48 29 c2             	sub    %rax,%rdx
  4035d9:	48 83 fa 0f          	cmp    $0xf,%rdx
  4035dd:	0f 8e 81 0a 00 00    	jle    404064 <__sprintf_chk@plt+0x17d4>
  4035e3:	48 8d 50 10          	lea    0x10(%rax),%rdx
  4035e7:	48 89 15 2a 79 21 00 	mov    %rdx,0x21792a(%rip)        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  4035ee:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
  4035f3:	48 89 50 08          	mov    %rdx,0x8(%rax)
  4035f7:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
  4035fc:	48 89 10             	mov    %rdx,(%rax)
  4035ff:	80 3d 08 7b 21 00 00 	cmpb   $0x0,0x217b08(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  403606:	75 0d                	jne    403615 <__sprintf_chk@plt+0xd85>
  403608:	80 3d c1 7a 21 00 00 	cmpb   $0x0,0x217ac1(%rip)        # 61b0d0 <stderr@@GLIBC_2.2.5+0xa80>
  40360f:	0f 84 c1 00 00 00    	je     4036d6 <__sprintf_chk@plt+0xe46>
  403615:	80 3d a4 6d 21 00 00 	cmpb   $0x0,0x216da4(%rip)        # 61a3c0 <_fini@@Base+0x2084c4>
  40361c:	75 28                	jne    403646 <__sprintf_chk@plt+0xdb6>
  40361e:	48 8b 3d eb 6f 21 00 	mov    0x216feb(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  403625:	48 8b 47 28          	mov    0x28(%rdi),%rax
  403629:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  40362d:	0f 83 b1 0f 00 00    	jae    4045e4 <__sprintf_chk@plt+0x1d54>
  403633:	48 8d 50 01          	lea    0x1(%rax),%rdx
  403637:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  40363b:	c6 00 0a             	movb   $0xa,(%rax)
  40363e:	48 83 05 d2 79 21 00 	addq   $0x1,0x2179d2(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403645:	01 
  403646:	80 3d e3 7a 21 00 00 	cmpb   $0x0,0x217ae3(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  40364d:	c6 05 6c 6d 21 00 00 	movb   $0x0,0x216d6c(%rip)        # 61a3c0 <_fini@@Base+0x2084c4>
  403654:	0f 85 d4 08 00 00    	jne    403f2e <__sprintf_chk@plt+0x169e>
  40365a:	48 85 db             	test   %rbx,%rbx
  40365d:	48 8b 15 7c 7a 21 00 	mov    0x217a7c(%rip),%rdx        # 61b0e0 <stderr@@GLIBC_2.2.5+0xa90>
  403664:	48 8b 3d a5 6f 21 00 	mov    0x216fa5(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  40366b:	49 0f 44 de          	cmove  %r14,%rbx
  40366f:	31 c9                	xor    %ecx,%ecx
  403671:	48 89 de             	mov    %rbx,%rsi
  403674:	e8 57 1c 00 00       	callq  4052d0 <__sprintf_chk@plt+0x2a40>
  403679:	48 01 05 98 79 21 00 	add    %rax,0x217998(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403680:	80 3d a9 7a 21 00 00 	cmpb   $0x0,0x217aa9(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  403687:	74 2a                	je     4036b3 <__sprintf_chk@plt+0xe23>
  403689:	48 8b 05 e8 78 21 00 	mov    0x2178e8(%rip),%rax        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  403690:	48 8d 50 08          	lea    0x8(%rax),%rdx
  403694:	48 39 15 e5 78 21 00 	cmp    %rdx,0x2178e5(%rip)        # 61af80 <stderr@@GLIBC_2.2.5+0x930>
  40369b:	0f 82 f9 09 00 00    	jb     40409a <__sprintf_chk@plt+0x180a>
  4036a1:	48 8b 15 70 79 21 00 	mov    0x217970(%rip),%rdx        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  4036a8:	48 89 10             	mov    %rdx,(%rax)
  4036ab:	48 83 05 c5 78 21 00 	addq   $0x8,0x2178c5(%rip)        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  4036b2:	08 
  4036b3:	48 8b 0d 56 6f 21 00 	mov    0x216f56(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  4036ba:	ba 02 00 00 00       	mov    $0x2,%edx
  4036bf:	be 01 00 00 00       	mov    $0x1,%esi
  4036c4:	bf 3b 39 41 00       	mov    $0x41393b,%edi
  4036c9:	e8 f2 ef ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  4036ce:	48 83 05 42 79 21 00 	addq   $0x2,0x217942(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  4036d5:	02 
  4036d6:	e8 f5 16 00 00       	callq  404dd0 <__sprintf_chk@plt+0x2540>
  4036db:	0f b6 44 24 2f       	movzbl 0x2f(%rsp),%eax
  4036e0:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
  4036e7:	00 00 
  4036e9:	89 44 24 10          	mov    %eax,0x10(%rsp)
  4036ed:	0f 1f 00             	nopl   (%rax)
  4036f0:	41 c7 04 24 00 00 00 	movl   $0x0,(%r12)
  4036f7:	00 
  4036f8:	4c 89 ef             	mov    %r13,%rdi
  4036fb:	e8 20 ef ff ff       	callq  402620 <readdir@plt>
  403700:	48 85 c0             	test   %rax,%rax
  403703:	48 89 c5             	mov    %rax,%rbp
  403706:	0f 84 b4 05 00 00    	je     403cc0 <__sprintf_chk@plt+0x1430>
  40370c:	48 8d 58 13          	lea    0x13(%rax),%rbx
  403710:	8b 05 f2 79 21 00    	mov    0x2179f2(%rip),%eax        # 61b108 <stderr@@GLIBC_2.2.5+0xab8>
  403716:	83 f8 02             	cmp    $0x2,%eax
  403719:	0f 84 59 05 00 00    	je     403c78 <__sprintf_chk@plt+0x13e8>
  40371f:	80 7d 13 2e          	cmpb   $0x2e,0x13(%rbp)
  403723:	0f 84 2f 05 00 00    	je     403c58 <__sprintf_chk@plt+0x13c8>
  403729:	85 c0                	test   %eax,%eax
  40372b:	0f 85 47 05 00 00    	jne    403c78 <__sprintf_chk@plt+0x13e8>
  403731:	4c 8b 3d c0 79 21 00 	mov    0x2179c0(%rip),%r15        # 61b0f8 <stderr@@GLIBC_2.2.5+0xaa8>
  403738:	4d 85 ff             	test   %r15,%r15
  40373b:	75 18                	jne    403755 <__sprintf_chk@plt+0xec5>
  40373d:	e9 36 05 00 00       	jmpq   403c78 <__sprintf_chk@plt+0x13e8>
  403742:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  403748:	4d 8b 7f 08          	mov    0x8(%r15),%r15
  40374c:	4d 85 ff             	test   %r15,%r15
  40374f:	0f 84 23 05 00 00    	je     403c78 <__sprintf_chk@plt+0x13e8>
  403755:	49 8b 3f             	mov    (%r15),%rdi
  403758:	ba 04 00 00 00       	mov    $0x4,%edx
  40375d:	48 89 de             	mov    %rbx,%rsi
  403760:	e8 0b ed ff ff       	callq  402470 <fnmatch@plt>
  403765:	85 c0                	test   %eax,%eax
  403767:	75 df                	jne    403748 <__sprintf_chk@plt+0xeb8>
  403769:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403770:	e8 1b 2d 00 00       	callq  406490 <__sprintf_chk@plt+0x3c00>
  403775:	e9 76 ff ff ff       	jmpq   4036f0 <__sprintf_chk@plt+0xe60>
  40377a:	c6 05 40 79 21 00 01 	movb   $0x1,0x217940(%rip)        # 61b0c1 <stderr@@GLIBC_2.2.5+0xa71>
  403781:	e9 18 fc ff ff       	jmpq   40339e <__sprintf_chk@plt+0xb0e>
  403786:	4c 89 e7             	mov    %r12,%rdi
  403789:	e8 a2 b1 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  40378e:	31 ff                	xor    %edi,%edi
  403790:	49 89 c4             	mov    %rax,%r12
  403793:	ba 05 00 00 00       	mov    $0x5,%edx
  403798:	be 00 5b 41 00       	mov    $0x415b00,%esi
  40379d:	e8 be eb ff ff       	callq  402360 <dcgettext@plt>
  4037a2:	4c 89 e1             	mov    %r12,%rcx
  4037a5:	48 89 c2             	mov    %rax,%rdx
  4037a8:	31 f6                	xor    %esi,%esi
  4037aa:	31 ff                	xor    %edi,%edi
  4037ac:	31 c0                	xor    %eax,%eax
  4037ae:	e8 bd ef ff ff       	callq  402770 <error@plt>
  4037b3:	e9 c7 f2 ff ff       	jmpq   402a7f <__sprintf_chk@plt+0x1ef>
  4037b8:	45 31 c0             	xor    %r8d,%r8d
  4037bb:	31 d2                	xor    %edx,%edx
  4037bd:	31 f6                	xor    %esi,%esi
  4037bf:	48 89 c1             	mov    %rax,%rcx
  4037c2:	4c 89 e7             	mov    %r12,%rdi
  4037c5:	e8 c6 d6 00 00       	callq  410e90 <__sprintf_chk@plt+0xe600>
  4037ca:	85 c0                	test   %eax,%eax
  4037cc:	0f 85 c1 02 00 00    	jne    403a93 <__sprintf_chk@plt+0x1203>
  4037d2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  4037d7:	48 85 c0             	test   %rax,%rax
  4037da:	0f 84 b3 02 00 00    	je     403a93 <__sprintf_chk@plt+0x1203>
  4037e0:	48 89 05 e1 78 21 00 	mov    %rax,0x2178e1(%rip)        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  4037e7:	e9 c5 f2 ff ff       	jmpq   402ab1 <__sprintf_chk@plt+0x221>
  4037ec:	c7 05 5a 79 21 00 01 	movl   $0x1,0x21795a(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4037f3:	00 00 00 
  4037f6:	e9 ca f1 ff ff       	jmpq   4029c5 <__sprintf_chk@plt+0x135>
  4037fb:	48 89 d0             	mov    %rdx,%rax
  4037fe:	b9 03 00 00 00       	mov    $0x3,%ecx
  403803:	31 d2                	xor    %edx,%edx
  403805:	48 f7 f1             	div    %rcx
  403808:	e9 35 fa ff ff       	jmpq   403242 <__sprintf_chk@plt+0x9b2>
  40380d:	4d 85 e4             	test   %r12,%r12
  403810:	0f 84 b0 0d 00 00    	je     4045c6 <__sprintf_chk@plt+0x1d36>
  403816:	41 be a0 2c 41 00    	mov    $0x412ca0,%r14d
  40381c:	41 bd 06 00 00 00    	mov    $0x6,%r13d
  403822:	eb 1a                	jmp    40383e <__sprintf_chk@plt+0xfae>
  403824:	0f 1f 40 00          	nopl   0x0(%rax)
  403828:	bf 02 00 00 00       	mov    $0x2,%edi
  40382d:	e8 3e 73 00 00       	callq  40ab70 <__sprintf_chk@plt+0x82e0>
  403832:	84 c0                	test   %al,%al
  403834:	0f 84 c7 fa ff ff    	je     403301 <__sprintf_chk@plt+0xa71>
  40383a:	49 83 c4 06          	add    $0x6,%r12
  40383e:	4c 89 e6             	mov    %r12,%rsi
  403841:	4c 89 f7             	mov    %r14,%rdi
  403844:	4c 89 e9             	mov    %r13,%rcx
  403847:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  403849:	74 dd                	je     403828 <__sprintf_chk@plt+0xf98>
  40384b:	41 80 3c 24 2b       	cmpb   $0x2b,(%r12)
  403850:	0f 84 83 0b 00 00    	je     4043d9 <__sprintf_chk@plt+0x1b49>
  403856:	b9 04 00 00 00       	mov    $0x4,%ecx
  40385b:	ba f0 36 41 00       	mov    $0x4136f0,%edx
  403860:	be 00 37 41 00       	mov    $0x413700,%esi
  403865:	4c 89 e7             	mov    %r12,%rdi
  403868:	e8 e3 65 00 00       	callq  409e50 <__sprintf_chk@plt+0x75c0>
  40386d:	48 85 c0             	test   %rax,%rax
  403870:	0f 88 ee 0c 00 00    	js     404564 <__sprintf_chk@plt+0x1cd4>
  403876:	48 83 f8 01          	cmp    $0x1,%rax
  40387a:	0f 84 c9 0c 00 00    	je     404549 <__sprintf_chk@plt+0x1cb9>
  403880:	0f 8e df 0b 00 00    	jle    404465 <__sprintf_chk@plt+0x1bd5>
  403886:	48 83 f8 02          	cmp    $0x2,%rax
  40388a:	0f 84 63 0d 00 00    	je     4045f3 <__sprintf_chk@plt+0x1d63>
  403890:	48 83 f8 03          	cmp    $0x3,%rax
  403894:	75 12                	jne    4038a8 <__sprintf_chk@plt+0x1018>
  403896:	bf 02 00 00 00       	mov    $0x2,%edi
  40389b:	e8 d0 72 00 00       	callq  40ab70 <__sprintf_chk@plt+0x82e0>
  4038a0:	84 c0                	test   %al,%al
  4038a2:	0f 85 66 0d 00 00    	jne    40460e <__sprintf_chk@plt+0x1d7e>
  4038a8:	48 8b 3d 21 6b 21 00 	mov    0x216b21(%rip),%rdi        # 61a3d0 <_fini@@Base+0x2084d4>
  4038af:	be 66 37 41 00       	mov    $0x413766,%esi
  4038b4:	e8 a7 ef ff ff       	callq  402860 <strstr@plt>
  4038b9:	48 85 c0             	test   %rax,%rax
  4038bc:	0f 84 c2 0b 00 00    	je     404484 <__sprintf_chk@plt+0x1bf4>
  4038c2:	48 c7 05 7b 6e 21 00 	movq   $0x5,0x216e7b(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  4038c9:	05 00 00 00 
  4038cd:	4c 8b 35 74 6e 21 00 	mov    0x216e74(%rip),%r14        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  4038d4:	41 bd 60 a7 61 00    	mov    $0x61a760,%r13d
  4038da:	48 c7 05 63 6e 21 00 	movq   $0x0,0x216e63(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  4038e1:	00 00 00 00 
  4038e5:	41 bc 0e 00 02 00    	mov    $0x2000e,%r12d
  4038eb:	44 89 e7             	mov    %r12d,%edi
  4038ee:	4c 89 74 24 40       	mov    %r14,0x40(%rsp)
  4038f3:	e8 68 ed ff ff       	callq  402660 <nl_langinfo@plt>
  4038f8:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  4038fd:	45 31 c9             	xor    %r9d,%r9d
  403900:	45 31 c0             	xor    %r8d,%r8d
  403903:	ba a1 00 00 00       	mov    $0xa1,%edx
  403908:	4c 89 ee             	mov    %r13,%rsi
  40390b:	48 89 c7             	mov    %rax,%rdi
  40390e:	e8 ad 94 00 00       	callq  40cdc0 <__sprintf_chk@plt+0xa530>
  403913:	48 3d a0 00 00 00    	cmp    $0xa0,%rax
  403919:	0f 87 37 0a 00 00    	ja     404356 <__sprintf_chk@plt+0x1ac6>
  40391f:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  403924:	48 39 05 1d 6e 21 00 	cmp    %rax,0x216e1d(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  40392b:	48 0f 43 05 15 6e 21 	cmovae 0x216e15(%rip),%rax        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  403932:	00 
  403933:	41 83 c4 01          	add    $0x1,%r12d
  403937:	49 81 c5 a1 00 00 00 	add    $0xa1,%r13
  40393e:	41 81 fc 1a 00 02 00 	cmp    $0x2001a,%r12d
  403945:	48 89 05 fc 6d 21 00 	mov    %rax,0x216dfc(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  40394c:	75 9d                	jne    4038eb <__sprintf_chk@plt+0x105b>
  40394e:	4c 39 f0             	cmp    %r14,%rax
  403951:	0f 82 76 ff ff ff    	jb     4038cd <__sprintf_chk@plt+0x103d>
  403957:	48 85 c0             	test   %rax,%rax
  40395a:	0f 85 a1 f9 ff ff    	jne    403301 <__sprintf_chk@plt+0xa71>
  403960:	e9 fc 09 00 00       	jmpq   404361 <__sprintf_chk@plt+0x1ad1>
  403965:	45 84 ed             	test   %r13b,%r13b
  403968:	0f 85 86 f9 ff ff    	jne    4032f4 <__sprintf_chk@plt+0xa64>
  40396e:	83 3d db 77 21 00 00 	cmpl   $0x0,0x2177db(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403975:	0f 84 92 fe ff ff    	je     40380d <__sprintf_chk@plt+0xf7d>
  40397b:	c7 05 c3 77 21 00 04 	movl   $0x4,0x2177c3(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  403982:	00 00 00 
  403985:	e9 6a f9 ff ff       	jmpq   4032f4 <__sprintf_chk@plt+0xa64>
  40398a:	bf e1 38 41 00       	mov    $0x4138e1,%edi
  40398f:	e8 2c e8 ff ff       	callq  4021c0 <getenv@plt>
  403994:	ba 38 b1 61 00       	mov    $0x61b138,%edx
  403999:	49 89 c7             	mov    %rax,%r15
  40399c:	be 40 b1 61 00       	mov    $0x61b140,%esi
  4039a1:	48 89 c7             	mov    %rax,%rdi
  4039a4:	e8 67 8e 00 00       	callq  40c810 <__sprintf_chk@plt+0x9f80>
  4039a9:	4d 85 ff             	test   %r15,%r15
  4039ac:	0f 84 7f 0b 00 00    	je     404531 <__sprintf_chk@plt+0x1ca1>
  4039b2:	8b 05 88 77 21 00    	mov    0x217788(%rip),%eax        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  4039b8:	89 05 76 77 21 00    	mov    %eax,0x217776(%rip)        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  4039be:	48 8b 05 73 77 21 00 	mov    0x217773(%rip),%rax        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  4039c5:	48 89 05 94 6b 21 00 	mov    %rax,0x216b94(%rip)        # 61a560 <_fini@@Base+0x208664>
  4039cc:	45 84 f6             	test   %r14b,%r14b
  4039cf:	0f 84 57 f8 ff ff    	je     40322c <__sprintf_chk@plt+0x99c>
  4039d5:	c7 05 61 77 21 00 00 	movl   $0x0,0x217761(%rip)        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  4039dc:	00 00 00 
  4039df:	48 c7 05 4e 77 21 00 	movq   $0x400,0x21774e(%rip)        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  4039e6:	00 04 00 00 
  4039ea:	e9 3d f8 ff ff       	jmpq   40322c <__sprintf_chk@plt+0x99c>
  4039ef:	31 f6                	xor    %esi,%esi
  4039f1:	41 b8 d0 49 40 00    	mov    $0x4049d0,%r8d
  4039f7:	b9 90 49 40 00       	mov    $0x404990,%ecx
  4039fc:	ba 80 49 40 00       	mov    $0x404980,%edx
  403a01:	bf 1e 00 00 00       	mov    $0x1e,%edi
  403a06:	e8 f5 79 00 00       	callq  40b400 <__sprintf_chk@plt+0x8b70>
  403a0b:	48 85 c0             	test   %rax,%rax
  403a0e:	48 89 05 b3 77 21 00 	mov    %rax,0x2177b3(%rip)        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  403a15:	0f 84 a0 09 00 00    	je     4043bb <__sprintf_chk@plt+0x1b2b>
  403a1b:	41 b8 f0 21 40 00    	mov    $0x4021f0,%r8d
  403a21:	b9 40 26 40 00       	mov    $0x402640,%ecx
  403a26:	31 d2                	xor    %edx,%edx
  403a28:	31 f6                	xor    %esi,%esi
  403a2a:	bf 00 af 61 00       	mov    $0x61af00,%edi
  403a2f:	e8 bc e9 ff ff       	callq  4023f0 <_obstack_begin@plt>
  403a34:	e9 f6 f8 ff ff       	jmpq   40332f <__sprintf_chk@plt+0xa9f>
  403a39:	80 3d cd 76 21 00 00 	cmpb   $0x0,0x2176cd(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  403a40:	b8 02 00 00 00       	mov    $0x2,%eax
  403a45:	75 18                	jne    403a5f <__sprintf_chk@plt+0x11cf>
  403a47:	83 3d de 76 21 00 03 	cmpl   $0x3,0x2176de(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  403a4e:	74 0f                	je     403a5f <__sprintf_chk@plt+0x11cf>
  403a50:	83 3d f9 76 21 00 01 	cmpl   $0x1,0x2176f9(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403a57:	19 c0                	sbb    %eax,%eax
  403a59:	83 e0 fe             	and    $0xfffffffe,%eax
  403a5c:	83 c0 04             	add    $0x4,%eax
  403a5f:	89 05 ab 76 21 00    	mov    %eax,0x2176ab(%rip)        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  403a65:	e9 b8 f8 ff ff       	jmpq   403322 <__sprintf_chk@plt+0xa92>
  403a6a:	bf 01 00 00 00       	mov    $0x1,%edi
  403a6f:	e8 0c e8 ff ff       	callq  402280 <isatty@plt>
  403a74:	85 c0                	test   %eax,%eax
  403a76:	0f 84 fc f5 ff ff    	je     403078 <__sprintf_chk@plt+0x7e8>
  403a7c:	c6 05 a6 76 21 00 01 	movb   $0x1,0x2176a6(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  403a83:	48 c7 05 4a 76 21 00 	movq   $0x0,0x21764a(%rip)        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  403a8a:	00 00 00 00 
  403a8e:	e9 9d f0 ff ff       	jmpq   402b30 <__sprintf_chk@plt+0x2a0>
  403a93:	4c 89 e7             	mov    %r12,%rdi
  403a96:	e8 95 ae 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  403a9b:	31 ff                	xor    %edi,%edi
  403a9d:	49 89 c4             	mov    %rax,%r12
  403aa0:	ba 05 00 00 00       	mov    $0x5,%edx
  403aa5:	be 48 5b 41 00       	mov    $0x415b48,%esi
  403aaa:	e8 b1 e8 ff ff       	callq  402360 <dcgettext@plt>
  403aaf:	4c 89 e1             	mov    %r12,%rcx
  403ab2:	48 89 c2             	mov    %rax,%rdx
  403ab5:	31 f6                	xor    %esi,%esi
  403ab7:	31 ff                	xor    %edi,%edi
  403ab9:	31 c0                	xor    %eax,%eax
  403abb:	e8 b0 ec ff ff       	callq  402770 <error@plt>
  403ac0:	e9 ec ef ff ff       	jmpq   402ab1 <__sprintf_chk@plt+0x221>
  403ac5:	bf 7f 39 41 00       	mov    $0x41397f,%edi
  403aca:	e8 f1 e6 ff ff       	callq  4021c0 <getenv@plt>
  403acf:	48 85 c0             	test   %rax,%rax
  403ad2:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  403ad7:	74 09                	je     403ae2 <__sprintf_chk@plt+0x1252>
  403ad9:	80 38 00             	cmpb   $0x0,(%rax)
  403adc:	0f 85 18 0a 00 00    	jne    4044fa <__sprintf_chk@plt+0x1c6a>
  403ae2:	80 3d 40 76 21 00 00 	cmpb   $0x0,0x217640(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  403ae9:	0f 84 26 f8 ff ff    	je     403315 <__sprintf_chk@plt+0xa85>
  403aef:	bf 0d 00 00 00       	mov    $0xd,%edi
  403af4:	e8 d7 11 00 00       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  403af9:	84 c0                	test   %al,%al
  403afb:	75 2e                	jne    403b2b <__sprintf_chk@plt+0x129b>
  403afd:	bf 0e 00 00 00       	mov    $0xe,%edi
  403b02:	e8 c9 11 00 00       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  403b07:	84 c0                	test   %al,%al
  403b09:	74 09                	je     403b14 <__sprintf_chk@plt+0x1284>
  403b0b:	80 3d 86 76 21 00 00 	cmpb   $0x0,0x217686(%rip)        # 61b198 <stderr@@GLIBC_2.2.5+0xb48>
  403b12:	75 17                	jne    403b2b <__sprintf_chk@plt+0x129b>
  403b14:	bf 0c 00 00 00       	mov    $0xc,%edi
  403b19:	e8 b2 11 00 00       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  403b1e:	84 c0                	test   %al,%al
  403b20:	74 10                	je     403b32 <__sprintf_chk@plt+0x12a2>
  403b22:	83 3d 27 76 21 00 00 	cmpl   $0x0,0x217627(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403b29:	75 07                	jne    403b32 <__sprintf_chk@plt+0x12a2>
  403b2b:	c6 05 e3 75 21 00 01 	movb   $0x1,0x2175e3(%rip)        # 61b115 <stderr@@GLIBC_2.2.5+0xac5>
  403b32:	bf 01 00 00 00       	mov    $0x1,%edi
  403b37:	e8 c4 ea ff ff       	callq  402600 <tcgetpgrp@plt>
  403b3c:	85 c0                	test   %eax,%eax
  403b3e:	0f 88 d1 f7 ff ff    	js     403315 <__sprintf_chk@plt+0xa85>
  403b44:	bf 40 b0 61 00       	mov    $0x61b040,%edi
  403b49:	45 31 ed             	xor    %r13d,%r13d
  403b4c:	e8 4f ea ff ff       	callq  4025a0 <sigemptyset@plt>
  403b51:	45 8b b5 c0 2c 41 00 	mov    0x412cc0(%r13),%r14d
  403b58:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  403b5d:	31 f6                	xor    %esi,%esi
  403b5f:	44 89 f7             	mov    %r14d,%edi
  403b62:	e8 29 e7 ff ff       	callq  402290 <sigaction@plt>
  403b67:	48 83 7c 24 40 01    	cmpq   $0x1,0x40(%rsp)
  403b6d:	74 0d                	je     403b7c <__sprintf_chk@plt+0x12ec>
  403b6f:	44 89 f6             	mov    %r14d,%esi
  403b72:	bf 40 b0 61 00       	mov    $0x61b040,%edi
  403b77:	e8 d4 ec ff ff       	callq  402850 <sigaddset@plt>
  403b7c:	49 83 c5 04          	add    $0x4,%r13
  403b80:	49 83 fd 30          	cmp    $0x30,%r13
  403b84:	75 cb                	jne    403b51 <__sprintf_chk@plt+0x12c1>
  403b86:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
  403b8b:	be 40 b0 61 00       	mov    $0x61b040,%esi
  403b90:	b9 20 00 00 00       	mov    $0x20,%ecx
  403b95:	f3 a5                	rep movsl %ds:(%rsi),%es:(%rdi)
  403b97:	c7 84 24 c8 00 00 00 	movl   $0x10000000,0xc8(%rsp)
  403b9e:	00 00 00 10 
  403ba2:	45 30 ed             	xor    %r13b,%r13b
  403ba5:	41 be b0 49 40 00    	mov    $0x4049b0,%r14d
  403bab:	45 8b bd c0 2c 41 00 	mov    0x412cc0(%r13),%r15d
  403bb2:	bf 40 b0 61 00       	mov    $0x61b040,%edi
  403bb7:	44 89 fe             	mov    %r15d,%esi
  403bba:	e8 21 ec ff ff       	callq  4027e0 <sigismember@plt>
  403bbf:	85 c0                	test   %eax,%eax
  403bc1:	74 21                	je     403be4 <__sprintf_chk@plt+0x1354>
  403bc3:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  403bc8:	41 83 ff 14          	cmp    $0x14,%r15d
  403bcc:	b8 f0 57 40 00       	mov    $0x4057f0,%eax
  403bd1:	49 0f 45 c6          	cmovne %r14,%rax
  403bd5:	44 89 ff             	mov    %r15d,%edi
  403bd8:	31 d2                	xor    %edx,%edx
  403bda:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  403bdf:	e8 ac e6 ff ff       	callq  402290 <sigaction@plt>
  403be4:	49 83 c5 04          	add    $0x4,%r13
  403be8:	49 83 fd 30          	cmp    $0x30,%r13
  403bec:	75 bd                	jne    403bab <__sprintf_chk@plt+0x131b>
  403bee:	e9 22 f7 ff ff       	jmpq   403315 <__sprintf_chk@plt+0xa85>
  403bf3:	be 07 38 41 00       	mov    $0x413807,%esi
  403bf8:	e9 29 f2 ff ff       	jmpq   402e26 <__sprintf_chk@plt+0x596>
  403bfd:	48 8b 3d 3c 6a 21 00 	mov    0x216a3c(%rip),%rdi        # 61a640 <optarg@@GLIBC_2.2.5>
  403c04:	e8 27 ad 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  403c09:	31 ff                	xor    %edi,%edi
  403c0b:	49 89 c7             	mov    %rax,%r15
  403c0e:	ba 05 00 00 00       	mov    $0x5,%edx
  403c13:	be 67 38 41 00       	mov    $0x413867,%esi
  403c18:	e8 43 e7 ff ff       	callq  402360 <dcgettext@plt>
  403c1d:	4c 89 f9             	mov    %r15,%rcx
  403c20:	48 89 c2             	mov    %rax,%rdx
  403c23:	31 f6                	xor    %esi,%esi
  403c25:	bf 02 00 00 00       	mov    $0x2,%edi
  403c2a:	31 c0                	xor    %eax,%eax
  403c2c:	e8 3f eb ff ff       	callq  402770 <error@plt>
  403c31:	e9 6f f0 ff ff       	jmpq   402ca5 <__sprintf_chk@plt+0x415>
  403c36:	bf 01 00 00 00       	mov    $0x1,%edi
  403c3b:	e8 40 e6 ff ff       	callq  402280 <isatty@plt>
  403c40:	83 f8 01             	cmp    $0x1,%eax
  403c43:	19 c0                	sbb    %eax,%eax
  403c45:	83 c0 02             	add    $0x2,%eax
  403c48:	89 05 02 75 21 00    	mov    %eax,0x217502(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403c4e:	e9 a8 ef ff ff       	jmpq   402bfb <__sprintf_chk@plt+0x36b>
  403c53:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  403c58:	85 c0                	test   %eax,%eax
  403c5a:	0f 84 10 fb ff ff    	je     403770 <__sprintf_chk@plt+0xee0>
  403c60:	31 c0                	xor    %eax,%eax
  403c62:	80 7d 14 2e          	cmpb   $0x2e,0x14(%rbp)
  403c66:	0f 94 c0             	sete   %al
  403c69:	80 7c 05 14 00       	cmpb   $0x0,0x14(%rbp,%rax,1)
  403c6e:	0f 84 fc fa ff ff    	je     403770 <__sprintf_chk@plt+0xee0>
  403c74:	0f 1f 40 00          	nopl   0x0(%rax)
  403c78:	4c 8b 3d 81 74 21 00 	mov    0x217481(%rip),%r15        # 61b100 <stderr@@GLIBC_2.2.5+0xab0>
  403c7f:	4d 85 ff             	test   %r15,%r15
  403c82:	75 19                	jne    403c9d <__sprintf_chk@plt+0x140d>
  403c84:	e9 97 01 00 00       	jmpq   403e20 <__sprintf_chk@plt+0x1590>
  403c89:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  403c90:	4d 8b 7f 08          	mov    0x8(%r15),%r15
  403c94:	4d 85 ff             	test   %r15,%r15
  403c97:	0f 84 83 01 00 00    	je     403e20 <__sprintf_chk@plt+0x1590>
  403c9d:	49 8b 3f             	mov    (%r15),%rdi
  403ca0:	ba 04 00 00 00       	mov    $0x4,%edx
  403ca5:	48 89 de             	mov    %rbx,%rsi
  403ca8:	e8 c3 e7 ff ff       	callq  402470 <fnmatch@plt>
  403cad:	85 c0                	test   %eax,%eax
  403caf:	75 df                	jne    403c90 <__sprintf_chk@plt+0x1400>
  403cb1:	e9 ba fa ff ff       	jmpq   403770 <__sprintf_chk@plt+0xee0>
  403cb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  403cbd:	00 00 00 
  403cc0:	41 8b 14 24          	mov    (%r12),%edx
  403cc4:	85 d2                	test   %edx,%edx
  403cc6:	74 2b                	je     403cf3 <__sprintf_chk@plt+0x1463>
  403cc8:	31 ff                	xor    %edi,%edi
  403cca:	ba 05 00 00 00       	mov    $0x5,%edx
  403ccf:	be b1 39 41 00       	mov    $0x4139b1,%esi
  403cd4:	e8 87 e6 ff ff       	callq  402360 <dcgettext@plt>
  403cd9:	8b 7c 24 10          	mov    0x10(%rsp),%edi
  403cdd:	4c 89 f2             	mov    %r14,%rdx
  403ce0:	48 89 c6             	mov    %rax,%rsi
  403ce3:	e8 28 1b 00 00       	callq  405810 <__sprintf_chk@plt+0x2f80>
  403ce8:	41 83 3c 24 4b       	cmpl   $0x4b,(%r12)
  403ced:	0f 84 7d fa ff ff    	je     403770 <__sprintf_chk@plt+0xee0>
  403cf3:	4c 89 ef             	mov    %r13,%rdi
  403cf6:	e8 e5 e7 ff ff       	callq  4024e0 <closedir@plt>
  403cfb:	85 c0                	test   %eax,%eax
  403cfd:	0f 85 b9 01 00 00    	jne    403ebc <__sprintf_chk@plt+0x162c>
  403d03:	e8 78 11 00 00       	callq  404e80 <__sprintf_chk@plt+0x25f0>
  403d08:	80 3d ff 73 21 00 00 	cmpb   $0x0,0x2173ff(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  403d0f:	0f 85 95 01 00 00    	jne    403eaa <__sprintf_chk@plt+0x161a>
  403d15:	8b 05 35 74 21 00    	mov    0x217435(%rip),%eax        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403d1b:	85 c0                	test   %eax,%eax
  403d1d:	74 0d                	je     403d2c <__sprintf_chk@plt+0x149c>
  403d1f:	80 3d 1e 74 21 00 00 	cmpb   $0x0,0x21741e(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  403d26:	0f 84 d4 00 00 00    	je     403e00 <__sprintf_chk@plt+0x1570>
  403d2c:	80 3d fd 73 21 00 00 	cmpb   $0x0,0x2173fd(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  403d33:	0f 85 54 02 00 00    	jne    403f8d <__sprintf_chk@plt+0x16fd>
  403d39:	ba 05 00 00 00       	mov    $0x5,%edx
  403d3e:	31 ff                	xor    %edi,%edi
  403d40:	be db 39 41 00       	mov    $0x4139db,%esi
  403d45:	e8 16 e6 ff ff       	callq  402360 <dcgettext@plt>
  403d4a:	48 8b 35 bf 68 21 00 	mov    0x2168bf(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  403d51:	48 89 c3             	mov    %rax,%rbx
  403d54:	48 89 c7             	mov    %rax,%rdi
  403d57:	e8 c4 e7 ff ff       	callq  402520 <fputs_unlocked@plt>
  403d5c:	48 89 df             	mov    %rbx,%rdi
  403d5f:	e8 1c e6 ff ff       	callq  402380 <strlen@plt>
  403d64:	48 8b 3d a5 68 21 00 	mov    0x2168a5(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  403d6b:	48 01 05 a6 72 21 00 	add    %rax,0x2172a6(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403d72:	48 8b 47 28          	mov    0x28(%rdi),%rax
  403d76:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  403d7a:	0f 83 6b 07 00 00    	jae    4044eb <__sprintf_chk@plt+0x1c5b>
  403d80:	48 8d 50 01          	lea    0x1(%rax),%rdx
  403d84:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  403d88:	c6 00 20             	movb   $0x20,(%rax)
  403d8b:	4c 8b 05 a6 73 21 00 	mov    0x2173a6(%rip),%r8        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  403d92:	8b 15 a8 73 21 00    	mov    0x2173a8(%rip),%edx        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  403d98:	48 8d b4 24 e0 00 00 	lea    0xe0(%rsp),%rsi
  403d9f:	00 
  403da0:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  403da5:	b9 00 02 00 00       	mov    $0x200,%ecx
  403daa:	48 83 05 66 72 21 00 	addq   $0x1,0x217266(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403db1:	01 
  403db2:	e8 b9 7f 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  403db7:	48 8b 35 52 68 21 00 	mov    0x216852(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  403dbe:	48 89 c3             	mov    %rax,%rbx
  403dc1:	48 89 c7             	mov    %rax,%rdi
  403dc4:	e8 57 e7 ff ff       	callq  402520 <fputs_unlocked@plt>
  403dc9:	48 89 df             	mov    %rbx,%rdi
  403dcc:	e8 af e5 ff ff       	callq  402380 <strlen@plt>
  403dd1:	48 8b 3d 38 68 21 00 	mov    0x216838(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  403dd8:	48 01 05 39 72 21 00 	add    %rax,0x217239(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403ddf:	48 8b 47 28          	mov    0x28(%rdi),%rax
  403de3:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  403de7:	0f 83 ef 06 00 00    	jae    4044dc <__sprintf_chk@plt+0x1c4c>
  403ded:	48 8d 50 01          	lea    0x1(%rax),%rdx
  403df1:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  403df5:	c6 00 0a             	movb   $0xa,(%rax)
  403df8:	48 83 05 18 72 21 00 	addq   $0x1,0x217218(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403dff:	01 
  403e00:	48 83 3d a8 73 21 00 	cmpq   $0x0,0x2173a8(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  403e07:	00 
  403e08:	0f 84 93 f6 ff ff    	je     4034a1 <__sprintf_chk@plt+0xc11>
  403e0e:	e8 dd 3b 00 00       	callq  4079f0 <__sprintf_chk@plt+0x5160>
  403e13:	e9 89 f6 ff ff       	jmpq   4034a1 <__sprintf_chk@plt+0xc11>
  403e18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403e1f:	00 
  403e20:	0f b6 45 12          	movzbl 0x12(%rbp),%eax
  403e24:	31 f6                	xor    %esi,%esi
  403e26:	83 e8 01             	sub    $0x1,%eax
  403e29:	3c 0d                	cmp    $0xd,%al
  403e2b:	77 0a                	ja     403e37 <__sprintf_chk@plt+0x15a7>
  403e2d:	0f b6 c0             	movzbl %al,%eax
  403e30:	8b 34 85 00 2c 41 00 	mov    0x412c00(,%rax,4),%esi
  403e37:	31 d2                	xor    %edx,%edx
  403e39:	4c 89 f1             	mov    %r14,%rcx
  403e3c:	48 89 df             	mov    %rbx,%rdi
  403e3f:	e8 5c 40 00 00       	callq  407ea0 <__sprintf_chk@plt+0x5610>
  403e44:	48 01 44 24 08       	add    %rax,0x8(%rsp)
  403e49:	83 3d 00 73 21 00 01 	cmpl   $0x1,0x217300(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  403e50:	0f 85 1a f9 ff ff    	jne    403770 <__sprintf_chk@plt+0xee0>
  403e56:	83 3d eb 72 21 00 ff 	cmpl   $0xffffffff,0x2172eb(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  403e5d:	0f 85 0d f9 ff ff    	jne    403770 <__sprintf_chk@plt+0xee0>
  403e63:	80 3d da 72 21 00 00 	cmpb   $0x0,0x2172da(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  403e6a:	0f 85 00 f9 ff ff    	jne    403770 <__sprintf_chk@plt+0xee0>
  403e70:	80 3d 97 72 21 00 00 	cmpb   $0x0,0x217297(%rip)        # 61b10e <stderr@@GLIBC_2.2.5+0xabe>
  403e77:	0f 85 f3 f8 ff ff    	jne    403770 <__sprintf_chk@plt+0xee0>
  403e7d:	e8 fe 0f 00 00       	callq  404e80 <__sprintf_chk@plt+0x25f0>
  403e82:	e8 69 3b 00 00       	callq  4079f0 <__sprintf_chk@plt+0x5160>
  403e87:	e8 44 0f 00 00       	callq  404dd0 <__sprintf_chk@plt+0x2540>
  403e8c:	0f 1f 40 00          	nopl   0x0(%rax)
  403e90:	e9 db f8 ff ff       	jmpq   403770 <__sprintf_chk@plt+0xee0>
  403e95:	4c 89 f6             	mov    %r14,%rsi
  403e98:	bf 01 00 00 00       	mov    $0x1,%edi
  403e9d:	e8 6e e7 ff ff       	callq  402610 <__xstat@plt>
  403ea2:	c1 e8 1f             	shr    $0x1f,%eax
  403ea5:	e9 c3 f6 ff ff       	jmpq   40356d <__sprintf_chk@plt+0xcdd>
  403eaa:	0f b6 74 24 2f       	movzbl 0x2f(%rsp),%esi
  403eaf:	4c 89 f7             	mov    %r14,%rdi
  403eb2:	e8 d9 11 00 00       	callq  405090 <__sprintf_chk@plt+0x2800>
  403eb7:	e9 59 fe ff ff       	jmpq   403d15 <__sprintf_chk@plt+0x1485>
  403ebc:	31 ff                	xor    %edi,%edi
  403ebe:	ba 05 00 00 00       	mov    $0x5,%edx
  403ec3:	be c6 39 41 00       	mov    $0x4139c6,%esi
  403ec8:	e8 93 e4 ff ff       	callq  402360 <dcgettext@plt>
  403ecd:	0f b6 7c 24 2f       	movzbl 0x2f(%rsp),%edi
  403ed2:	4c 89 f2             	mov    %r14,%rdx
  403ed5:	48 89 c6             	mov    %rax,%rsi
  403ed8:	e8 33 19 00 00       	callq  405810 <__sprintf_chk@plt+0x2f80>
  403edd:	e9 21 fe ff ff       	jmpq   403d03 <__sprintf_chk@plt+0x1473>
  403ee2:	48 89 ef             	mov    %rbp,%rdi
  403ee5:	e8 06 e3 ff ff       	callq  4021f0 <free@plt>
  403eea:	4c 89 f7             	mov    %r14,%rdi
  403eed:	e8 be ab 00 00       	callq  40eab0 <__sprintf_chk@plt+0xc220>
  403ef2:	ba 05 00 00 00       	mov    $0x5,%edx
  403ef7:	48 89 c3             	mov    %rax,%rbx
  403efa:	be 10 5d 41 00       	mov    $0x415d10,%esi
  403eff:	31 ff                	xor    %edi,%edi
  403f01:	e8 5a e4 ff ff       	callq  402360 <dcgettext@plt>
  403f06:	48 89 d9             	mov    %rbx,%rcx
  403f09:	48 89 c2             	mov    %rax,%rdx
  403f0c:	31 f6                	xor    %esi,%esi
  403f0e:	31 ff                	xor    %edi,%edi
  403f10:	31 c0                	xor    %eax,%eax
  403f12:	e8 59 e8 ff ff       	callq  402770 <error@plt>
  403f17:	4c 89 ef             	mov    %r13,%rdi
  403f1a:	e8 c1 e5 ff ff       	callq  4024e0 <closedir@plt>
  403f1f:	c7 05 07 71 21 00 02 	movl   $0x2,0x217107(%rip)        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  403f26:	00 00 00 
  403f29:	e9 73 f5 ff ff       	jmpq   4034a1 <__sprintf_chk@plt+0xc11>
  403f2e:	48 8b 0d db 66 21 00 	mov    0x2166db(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  403f35:	ba 02 00 00 00       	mov    $0x2,%edx
  403f3a:	be 01 00 00 00       	mov    $0x1,%esi
  403f3f:	bf 71 37 41 00       	mov    $0x413771,%edi
  403f44:	e8 77 e7 ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  403f49:	48 83 05 c7 70 21 00 	addq   $0x2,0x2170c7(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403f50:	02 
  403f51:	80 3d d8 71 21 00 00 	cmpb   $0x0,0x2171d8(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  403f58:	0f 84 fc f6 ff ff    	je     40365a <__sprintf_chk@plt+0xdca>
  403f5e:	48 8b 05 13 70 21 00 	mov    0x217013(%rip),%rax        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  403f65:	48 8d 50 08          	lea    0x8(%rax),%rdx
  403f69:	48 39 15 10 70 21 00 	cmp    %rdx,0x217010(%rip)        # 61af80 <stderr@@GLIBC_2.2.5+0x930>
  403f70:	0f 82 09 01 00 00    	jb     40407f <__sprintf_chk@plt+0x17ef>
  403f76:	48 8b 15 9b 70 21 00 	mov    0x21709b(%rip),%rdx        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403f7d:	48 89 10             	mov    %rdx,(%rax)
  403f80:	48 83 05 f0 6f 21 00 	addq   $0x8,0x216ff0(%rip)        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  403f87:	08 
  403f88:	e9 cd f6 ff ff       	jmpq   40365a <__sprintf_chk@plt+0xdca>
  403f8d:	48 8b 0d 7c 66 21 00 	mov    0x21667c(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  403f94:	ba 02 00 00 00       	mov    $0x2,%edx
  403f99:	be 01 00 00 00       	mov    $0x1,%esi
  403f9e:	bf 71 37 41 00       	mov    $0x413771,%edi
  403fa3:	e8 18 e7 ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  403fa8:	48 83 05 68 70 21 00 	addq   $0x2,0x217068(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  403faf:	02 
  403fb0:	e9 84 fd ff ff       	jmpq   403d39 <__sprintf_chk@plt+0x14a9>
  403fb5:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  403fba:	4c 8b 30             	mov    (%rax),%r14
  403fbd:	e9 46 f5 ff ff       	jmpq   403508 <__sprintf_chk@plt+0xc78>
  403fc2:	48 8b 05 4f 6f 21 00 	mov    0x216f4f(%rip),%rax        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  403fc9:	48 89 c2             	mov    %rax,%rdx
  403fcc:	48 2b 15 3d 6f 21 00 	sub    0x216f3d(%rip),%rdx        # 61af10 <stderr@@GLIBC_2.2.5+0x8c0>
  403fd3:	83 fa 0f             	cmp    $0xf,%edx
  403fd6:	0f 86 e7 04 00 00    	jbe    4044c3 <__sprintf_chk@plt+0x1c33>
  403fdc:	48 8b 15 3d 6f 21 00 	mov    0x216f3d(%rip),%rdx        # 61af20 <stderr@@GLIBC_2.2.5+0x8d0>
  403fe3:	48 29 c2             	sub    %rax,%rdx
  403fe6:	48 83 fa f0          	cmp    $0xfffffffffffffff0,%rdx
  403fea:	7d 16                	jge    404002 <__sprintf_chk@plt+0x1772>
  403fec:	be f0 ff ff ff       	mov    $0xfffffff0,%esi
  403ff1:	bf 00 af 61 00       	mov    $0x61af00,%edi
  403ff6:	e8 25 e7 ff ff       	callq  402720 <_obstack_newchunk@plt>
  403ffb:	48 8b 05 16 6f 21 00 	mov    0x216f16(%rip),%rax        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  404002:	48 8d 50 f0          	lea    -0x10(%rax),%rdx
  404006:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  40400b:	48 8b 3d b6 71 21 00 	mov    0x2171b6(%rip),%rdi        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  404012:	48 89 15 ff 6e 21 00 	mov    %rdx,0x216eff(%rip)        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  404019:	48 8b 50 f0          	mov    -0x10(%rax),%rdx
  40401d:	48 8b 40 f8          	mov    -0x8(%rax),%rax
  404021:	48 89 54 24 40       	mov    %rdx,0x40(%rsp)
  404026:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40402b:	e8 60 7b 00 00       	callq  40bb90 <__sprintf_chk@plt+0x9300>
  404030:	48 85 c0             	test   %rax,%rax
  404033:	0f 84 4e 02 00 00    	je     404287 <__sprintf_chk@plt+0x19f7>
  404039:	48 89 c7             	mov    %rax,%rdi
  40403c:	e8 af e1 ff ff       	callq  4021f0 <free@plt>
  404041:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
  404046:	48 8b 3b             	mov    (%rbx),%rdi
  404049:	e8 a2 e1 ff ff       	callq  4021f0 <free@plt>
  40404e:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  404052:	e8 99 e1 ff ff       	callq  4021f0 <free@plt>
  404057:	48 89 df             	mov    %rbx,%rdi
  40405a:	e8 91 e1 ff ff       	callq  4021f0 <free@plt>
  40405f:	e9 62 f4 ff ff       	jmpq   4034c6 <__sprintf_chk@plt+0xc36>
  404064:	be 10 00 00 00       	mov    $0x10,%esi
  404069:	bf 00 af 61 00       	mov    $0x61af00,%edi
  40406e:	e8 ad e6 ff ff       	callq  402720 <_obstack_newchunk@plt>
  404073:	48 8b 05 9e 6e 21 00 	mov    0x216e9e(%rip),%rax        # 61af18 <stderr@@GLIBC_2.2.5+0x8c8>
  40407a:	e9 64 f5 ff ff       	jmpq   4035e3 <__sprintf_chk@plt+0xd53>
  40407f:	be 08 00 00 00       	mov    $0x8,%esi
  404084:	bf 60 af 61 00       	mov    $0x61af60,%edi
  404089:	e8 92 e6 ff ff       	callq  402720 <_obstack_newchunk@plt>
  40408e:	48 8b 05 e3 6e 21 00 	mov    0x216ee3(%rip),%rax        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  404095:	e9 dc fe ff ff       	jmpq   403f76 <__sprintf_chk@plt+0x16e6>
  40409a:	be 08 00 00 00       	mov    $0x8,%esi
  40409f:	bf 60 af 61 00       	mov    $0x61af60,%edi
  4040a4:	e8 77 e6 ff ff       	callq  402720 <_obstack_newchunk@plt>
  4040a9:	48 8b 05 c8 6e 21 00 	mov    0x216ec8(%rip),%rax        # 61af78 <stderr@@GLIBC_2.2.5+0x928>
  4040b0:	e9 ec f5 ff ff       	jmpq   4036a1 <__sprintf_chk@plt+0xe11>
  4040b5:	80 3d 6d 70 21 00 00 	cmpb   $0x0,0x21706d(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  4040bc:	0f 84 90 00 00 00    	je     404152 <__sprintf_chk@plt+0x18c2>
  4040c2:	80 3d 5f 70 21 00 00 	cmpb   $0x0,0x21705f(%rip)        # 61b128 <stderr@@GLIBC_2.2.5+0xad8>
  4040c9:	74 22                	je     4040ed <__sprintf_chk@plt+0x185d>
  4040cb:	48 83 3d 0d 63 21 00 	cmpq   $0x2,0x21630d(%rip)        # 61a3e0 <_fini@@Base+0x2084e4>
  4040d2:	02 
  4040d3:	0f 84 f2 01 00 00    	je     4042cb <__sprintf_chk@plt+0x1a3b>
  4040d9:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  4040de:	e8 5d 23 00 00       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4040e3:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  4040e8:	e8 53 23 00 00       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4040ed:	48 8b 3d 1c 65 21 00 	mov    0x21651c(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  4040f4:	bb c0 2c 41 00       	mov    $0x412cc0,%ebx
  4040f9:	e8 22 e7 ff ff       	callq  402820 <fflush_unlocked@plt>
  4040fe:	eb 0d                	jmp    40410d <__sprintf_chk@plt+0x187d>
  404100:	48 83 c3 04          	add    $0x4,%rbx
  404104:	48 81 fb f0 2c 41 00 	cmp    $0x412cf0,%rbx
  40410b:	74 1d                	je     40412a <__sprintf_chk@plt+0x189a>
  40410d:	8b 2b                	mov    (%rbx),%ebp
  40410f:	bf 40 b0 61 00       	mov    $0x61b040,%edi
  404114:	89 ee                	mov    %ebp,%esi
  404116:	e8 c5 e6 ff ff       	callq  4027e0 <sigismember@plt>
  40411b:	85 c0                	test   %eax,%eax
  40411d:	74 e1                	je     404100 <__sprintf_chk@plt+0x1870>
  40411f:	31 f6                	xor    %esi,%esi
  404121:	89 ef                	mov    %ebp,%edi
  404123:	e8 38 e4 ff ff       	callq  402560 <signal@plt>
  404128:	eb d6                	jmp    404100 <__sprintf_chk@plt+0x1870>
  40412a:	8b 1d 04 6f 21 00    	mov    0x216f04(%rip),%ebx        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  404130:	85 db                	test   %ebx,%ebx
  404132:	74 0f                	je     404143 <__sprintf_chk@plt+0x18b3>
  404134:	bf 13 00 00 00       	mov    $0x13,%edi
  404139:	e8 a2 e0 ff ff       	callq  4021e0 <raise@plt>
  40413e:	83 eb 01             	sub    $0x1,%ebx
  404141:	75 f1                	jne    404134 <__sprintf_chk@plt+0x18a4>
  404143:	8b 3d ef 6e 21 00    	mov    0x216eef(%rip),%edi        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  404149:	85 ff                	test   %edi,%edi
  40414b:	74 05                	je     404152 <__sprintf_chk@plt+0x18c2>
  40414d:	e8 8e e0 ff ff       	callq  4021e0 <raise@plt>
  404152:	80 3d d7 6f 21 00 00 	cmpb   $0x0,0x216fd7(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  404159:	0f 85 ad 01 00 00    	jne    40430c <__sprintf_chk@plt+0x1a7c>
  40415f:	48 8b 1d 62 70 21 00 	mov    0x217062(%rip),%rbx        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  404166:	48 85 db             	test   %rbx,%rbx
  404169:	0f 84 86 00 00 00    	je     4041f5 <__sprintf_chk@plt+0x1965>
  40416f:	48 89 df             	mov    %rbx,%rdi
  404172:	e8 39 6e 00 00       	callq  40afb0 <__sprintf_chk@plt+0x8720>
  404177:	48 85 c0             	test   %rax,%rax
  40417a:	74 71                	je     4041ed <__sprintf_chk@plt+0x195d>
  40417c:	b9 a7 2c 41 00       	mov    $0x412ca7,%ecx
  404181:	ba dc 05 00 00       	mov    $0x5dc,%edx
  404186:	be 36 37 41 00       	mov    $0x413736,%esi
  40418b:	bf 68 5d 41 00       	mov    $0x415d68,%edi
  404190:	e8 bb e2 ff ff       	callq  402450 <__assert_fail@plt>
  404195:	31 ff                	xor    %edi,%edi
  404197:	ba 05 00 00 00       	mov    $0x5,%edx
  40419c:	be 98 39 41 00       	mov    $0x413998,%esi
  4041a1:	e8 ba e1 ff ff       	callq  402360 <dcgettext@plt>
  4041a6:	0f b6 7c 24 2f       	movzbl 0x2f(%rsp),%edi
  4041ab:	4c 89 f2             	mov    %r14,%rdx
  4041ae:	48 89 c6             	mov    %rax,%rsi
  4041b1:	e8 5a 16 00 00       	callq  405810 <__sprintf_chk@plt+0x2f80>
  4041b6:	e9 e6 f2 ff ff       	jmpq   4034a1 <__sprintf_chk@plt+0xc11>
  4041bb:	4c 89 e7             	mov    %r12,%rdi
  4041be:	e8 6d a7 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  4041c3:	31 ff                	xor    %edi,%edi
  4041c5:	49 89 c4             	mov    %rax,%r12
  4041c8:	ba 05 00 00 00       	mov    $0x5,%edx
  4041cd:	be 88 5b 41 00       	mov    $0x415b88,%esi
  4041d2:	e8 89 e1 ff ff       	callq  402360 <dcgettext@plt>
  4041d7:	4c 89 e1             	mov    %r12,%rcx
  4041da:	48 89 c2             	mov    %rax,%rdx
  4041dd:	31 f6                	xor    %esi,%esi
  4041df:	31 ff                	xor    %edi,%edi
  4041e1:	31 c0                	xor    %eax,%eax
  4041e3:	e8 88 e5 ff ff       	callq  402770 <error@plt>
  4041e8:	e9 35 e9 ff ff       	jmpq   402b22 <__sprintf_chk@plt+0x292>
  4041ed:	48 89 df             	mov    %rbx,%rdi
  4041f0:	e8 4b 74 00 00       	callq  40b640 <__sprintf_chk@plt+0x8db0>
  4041f5:	8b 3d 35 6e 21 00    	mov    0x216e35(%rip),%edi        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  4041fb:	e8 f0 e5 ff ff       	callq  4027f0 <exit@plt>
  404200:	e8 7b 0c 00 00       	callq  404e80 <__sprintf_chk@plt+0x25f0>
  404205:	80 3d 01 6f 21 00 00 	cmpb   $0x0,0x216f01(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  40420c:	0f 84 42 02 00 00    	je     404454 <__sprintf_chk@plt+0x1bc4>
  404212:	48 83 3d 96 6f 21 00 	cmpq   $0x0,0x216f96(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404219:	00 
  40421a:	0f 84 3b f2 ff ff    	je     40345b <__sprintf_chk@plt+0xbcb>
  404220:	e8 cb 37 00 00       	callq  4079f0 <__sprintf_chk@plt+0x5160>
  404225:	48 83 3d 63 6f 21 00 	cmpq   $0x0,0x216f63(%rip)        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  40422c:	00 
  40422d:	0f 84 82 02 00 00    	je     4044b5 <__sprintf_chk@plt+0x1c25>
  404233:	48 8b 3d d6 63 21 00 	mov    0x2163d6(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  40423a:	48 8b 47 28          	mov    0x28(%rdi),%rax
  40423e:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  404242:	0f 83 5b 02 00 00    	jae    4044a3 <__sprintf_chk@plt+0x1c13>
  404248:	48 8d 50 01          	lea    0x1(%rax),%rdx
  40424c:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  404250:	c6 00 0a             	movb   $0xa,(%rax)
  404253:	48 8b 05 36 6f 21 00 	mov    0x216f36(%rip),%rax        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  40425a:	48 83 05 b6 6d 21 00 	addq   $0x1,0x216db6(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  404261:	01 
  404262:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  404267:	e9 66 f2 ff ff       	jmpq   4034d2 <__sprintf_chk@plt+0xc42>
  40426c:	48 8b 3d 75 6e 21 00 	mov    0x216e75(%rip),%rdi        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  404273:	ba 01 00 00 00       	mov    $0x1,%edx
  404278:	be 20 00 00 00       	mov    $0x20,%esi
  40427d:	e8 ce a3 00 00       	callq  40e650 <__sprintf_chk@plt+0xbdc0>
  404282:	e9 e1 ef ff ff       	jmpq   403268 <__sprintf_chk@plt+0x9d8>
  404287:	b9 a7 2c 41 00       	mov    $0x412ca7,%ecx
  40428c:	ba 9d 05 00 00       	mov    $0x59d,%edx
  404291:	be 36 37 41 00       	mov    $0x413736,%esi
  404296:	bf 92 39 41 00       	mov    $0x413992,%edi
  40429b:	e8 b0 e1 ff ff       	callq  402450 <__assert_fail@plt>
  4042a0:	80 3d 66 6e 21 00 00 	cmpb   $0x0,0x216e66(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  4042a7:	0f 84 f8 00 00 00    	je     4043a5 <__sprintf_chk@plt+0x1b15>
  4042ad:	b9 19 69 41 00       	mov    $0x416919,%ecx
  4042b2:	ba 01 00 00 00       	mov    $0x1,%edx
  4042b7:	be 03 00 00 00       	mov    $0x3,%esi
  4042bc:	bf 90 39 41 00       	mov    $0x413990,%edi
  4042c1:	e8 da 3b 00 00       	callq  407ea0 <__sprintf_chk@plt+0x5610>
  4042c6:	e9 82 f1 ff ff       	jmpq   40344d <__sprintf_chk@plt+0xbbd>
  4042cb:	48 8b 3d 16 61 21 00 	mov    0x216116(%rip),%rdi        # 61a3e8 <_fini@@Base+0x2084ec>
  4042d2:	ba 02 00 00 00       	mov    $0x2,%edx
  4042d7:	be e1 39 41 00       	mov    $0x4139e1,%esi
  4042dc:	e8 1f e2 ff ff       	callq  402500 <memcmp@plt>
  4042e1:	85 c0                	test   %eax,%eax
  4042e3:	0f 85 f0 fd ff ff    	jne    4040d9 <__sprintf_chk@plt+0x1849>
  4042e9:	48 83 3d ff 60 21 00 	cmpq   $0x1,0x2160ff(%rip)        # 61a3f0 <_fini@@Base+0x2084f4>
  4042f0:	01 
  4042f1:	0f 85 e2 fd ff ff    	jne    4040d9 <__sprintf_chk@plt+0x1849>
  4042f7:	48 8b 05 fa 60 21 00 	mov    0x2160fa(%rip),%rax        # 61a3f8 <_fini@@Base+0x2084fc>
  4042fe:	80 38 6d             	cmpb   $0x6d,(%rax)
  404301:	0f 85 d2 fd ff ff    	jne    4040d9 <__sprintf_chk@plt+0x1849>
  404307:	e9 e1 fd ff ff       	jmpq   4040ed <__sprintf_chk@plt+0x185d>
  40430c:	be c0 af 61 00       	mov    $0x61afc0,%esi
  404311:	bf e4 39 41 00       	mov    $0x4139e4,%edi
  404316:	e8 15 13 00 00       	callq  405630 <__sprintf_chk@plt+0x2da0>
  40431b:	be 60 af 61 00       	mov    $0x61af60,%esi
  404320:	bf ee 39 41 00       	mov    $0x4139ee,%edi
  404325:	e8 06 13 00 00       	callq  405630 <__sprintf_chk@plt+0x2da0>
  40432a:	48 8b 3d b7 6d 21 00 	mov    0x216db7(%rip),%rdi        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  404331:	e8 fa a2 00 00       	callq  40e630 <__sprintf_chk@plt+0xbda0>
  404336:	89 c0                	mov    %eax,%eax
  404338:	be 40 5d 41 00       	mov    $0x415d40,%esi
  40433d:	bf 01 00 00 00       	mov    $0x1,%edi
  404342:	48 8b 14 c5 80 64 41 	mov    0x416480(,%rax,8),%rdx
  404349:	00 
  40434a:	31 c0                	xor    %eax,%eax
  40434c:	e8 df e3 ff ff       	callq  402730 <__printf_chk@plt>
  404351:	e9 09 fe ff ff       	jmpq   40415f <__sprintf_chk@plt+0x18cf>
  404356:	48 c7 05 e7 63 21 00 	movq   $0x0,0x2163e7(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  40435d:	00 00 00 00 
  404361:	31 ff                	xor    %edi,%edi
  404363:	ba 05 00 00 00       	mov    $0x5,%edx
  404368:	be 30 5c 41 00       	mov    $0x415c30,%esi
  40436d:	e8 ee df ff ff       	callq  402360 <dcgettext@plt>
  404372:	31 f6                	xor    %esi,%esi
  404374:	48 89 c2             	mov    %rax,%rdx
  404377:	31 ff                	xor    %edi,%edi
  404379:	31 c0                	xor    %eax,%eax
  40437b:	e8 f0 e3 ff ff       	callq  402770 <error@plt>
  404380:	e9 7c ef ff ff       	jmpq   403301 <__sprintf_chk@plt+0xa71>
  404385:	48 85 c0             	test   %rax,%rax
  404388:	0f 84 44 f1 ff ff    	je     4034d2 <__sprintf_chk@plt+0xc42>
  40438e:	48 83 78 18 00       	cmpq   $0x0,0x18(%rax)
  404393:	0f 85 39 f1 ff ff    	jne    4034d2 <__sprintf_chk@plt+0xc42>
  404399:	c6 05 30 6d 21 00 00 	movb   $0x0,0x216d30(%rip)        # 61b0d0 <stderr@@GLIBC_2.2.5+0xa80>
  4043a0:	e9 2d f1 ff ff       	jmpq   4034d2 <__sprintf_chk@plt+0xc42>
  4043a5:	ba 01 00 00 00       	mov    $0x1,%edx
  4043aa:	31 f6                	xor    %esi,%esi
  4043ac:	bf 90 39 41 00       	mov    $0x413990,%edi
  4043b1:	e8 6a 09 00 00       	callq  404d20 <__sprintf_chk@plt+0x2490>
  4043b6:	e9 92 f0 ff ff       	jmpq   40344d <__sprintf_chk@plt+0xbbd>
  4043bb:	e8 90 ca 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  4043c0:	4c 8b 05 79 62 21 00 	mov    0x216279(%rip),%r8        # 61a640 <optarg@@GLIBC_2.2.5>
  4043c7:	8b 74 24 38          	mov    0x38(%rsp),%esi
  4043cb:	b9 80 30 41 00       	mov    $0x413080,%ecx
  4043d0:	31 d2                	xor    %edx,%edx
  4043d2:	89 c7                	mov    %eax,%edi
  4043d4:	e8 f7 ce 00 00       	callq  4112d0 <__sprintf_chk@plt+0xea40>
  4043d9:	49 83 c4 01          	add    $0x1,%r12
  4043dd:	be 0a 00 00 00       	mov    $0xa,%esi
  4043e2:	4c 89 e7             	mov    %r12,%rdi
  4043e5:	e8 e6 df ff ff       	callq  4023d0 <strchr@plt>
  4043ea:	48 85 c0             	test   %rax,%rax
  4043ed:	49 89 c6             	mov    %rax,%r14
  4043f0:	74 5d                	je     40444f <__sprintf_chk@plt+0x1bbf>
  4043f2:	4c 8d 68 01          	lea    0x1(%rax),%r13
  4043f6:	be 0a 00 00 00       	mov    $0xa,%esi
  4043fb:	4c 89 ef             	mov    %r13,%rdi
  4043fe:	e8 cd df ff ff       	callq  4023d0 <strchr@plt>
  404403:	48 85 c0             	test   %rax,%rax
  404406:	74 30                	je     404438 <__sprintf_chk@plt+0x1ba8>
  404408:	4c 89 e7             	mov    %r12,%rdi
  40440b:	e8 00 a8 00 00       	callq  40ec10 <__sprintf_chk@plt+0xc380>
  404410:	ba 05 00 00 00       	mov    $0x5,%edx
  404415:	49 89 c7             	mov    %rax,%r15
  404418:	be 00 39 41 00       	mov    $0x413900,%esi
  40441d:	31 ff                	xor    %edi,%edi
  40441f:	e8 3c df ff ff       	callq  402360 <dcgettext@plt>
  404424:	4c 89 f9             	mov    %r15,%rcx
  404427:	48 89 c2             	mov    %rax,%rdx
  40442a:	31 f6                	xor    %esi,%esi
  40442c:	bf 02 00 00 00       	mov    $0x2,%edi
  404431:	31 c0                	xor    %eax,%eax
  404433:	e8 38 e3 ff ff       	callq  402770 <error@plt>
  404438:	41 c6 06 00          	movb   $0x0,(%r14)
  40443c:	4c 89 25 8d 5f 21 00 	mov    %r12,0x215f8d(%rip)        # 61a3d0 <_fini@@Base+0x2084d4>
  404443:	4c 89 2d 8e 5f 21 00 	mov    %r13,0x215f8e(%rip)        # 61a3d8 <_fini@@Base+0x2084dc>
  40444a:	e9 59 f4 ff ff       	jmpq   4038a8 <__sprintf_chk@plt+0x1018>
  40444f:	4d 89 e5             	mov    %r12,%r13
  404452:	eb e8                	jmp    40443c <__sprintf_chk@plt+0x1bac>
  404454:	be 01 00 00 00       	mov    $0x1,%esi
  404459:	31 ff                	xor    %edi,%edi
  40445b:	e8 30 0c 00 00       	callq  405090 <__sprintf_chk@plt+0x2800>
  404460:	e9 ad fd ff ff       	jmpq   404212 <__sprintf_chk@plt+0x1982>
  404465:	48 85 c0             	test   %rax,%rax
  404468:	0f 85 3a f4 ff ff    	jne    4038a8 <__sprintf_chk@plt+0x1018>
  40446e:	48 c7 05 5f 5f 21 00 	movq   $0x41394e,0x215f5f(%rip)        # 61a3d8 <_fini@@Base+0x2084dc>
  404475:	4e 39 41 00 
  404479:	48 c7 05 4c 5f 21 00 	movq   $0x41394e,0x215f4c(%rip)        # 61a3d0 <_fini@@Base+0x2084d4>
  404480:	4e 39 41 00 
  404484:	48 8b 3d 4d 5f 21 00 	mov    0x215f4d(%rip),%rdi        # 61a3d8 <_fini@@Base+0x2084dc>
  40448b:	be 66 37 41 00       	mov    $0x413766,%esi
  404490:	e8 cb e3 ff ff       	callq  402860 <strstr@plt>
  404495:	48 85 c0             	test   %rax,%rax
  404498:	0f 85 24 f4 ff ff    	jne    4038c2 <__sprintf_chk@plt+0x1032>
  40449e:	e9 5e ee ff ff       	jmpq   403301 <__sprintf_chk@plt+0xa71>
  4044a3:	be 0a 00 00 00       	mov    $0xa,%esi
  4044a8:	e8 53 df ff ff       	callq  402400 <__overflow@plt>
  4044ad:	0f 1f 00             	nopl   (%rax)
  4044b0:	e9 9e fd ff ff       	jmpq   404253 <__sprintf_chk@plt+0x19c3>
  4044b5:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  4044bc:	00 00 
  4044be:	e9 0f f0 ff ff       	jmpq   4034d2 <__sprintf_chk@plt+0xc42>
  4044c3:	b9 38 2c 41 00       	mov    $0x412c38,%ecx
  4044c8:	ba d5 03 00 00       	mov    $0x3d5,%edx
  4044cd:	be 36 37 41 00       	mov    $0x413736,%esi
  4044d2:	bf 58 5c 41 00       	mov    $0x415c58,%edi
  4044d7:	e8 74 df ff ff       	callq  402450 <__assert_fail@plt>
  4044dc:	be 0a 00 00 00       	mov    $0xa,%esi
  4044e1:	e8 1a df ff ff       	callq  402400 <__overflow@plt>
  4044e6:	e9 0d f9 ff ff       	jmpq   403df8 <__sprintf_chk@plt+0x1568>
  4044eb:	be 20 00 00 00       	mov    $0x20,%esi
  4044f0:	e8 0b df ff ff       	callq  402400 <__overflow@plt>
  4044f5:	e9 91 f8 ff ff       	jmpq   403d8b <__sprintf_chk@plt+0x14fb>
  4044fa:	48 89 c7             	mov    %rax,%rdi
  4044fd:	66 c7 44 24 30 3f 3f 	movw   $0x3f3f,0x30(%rsp)
  404504:	c6 44 24 32 00       	movb   $0x0,0x32(%rsp)
  404509:	45 31 ed             	xor    %r13d,%r13d
  40450c:	e8 1f c9 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  404511:	31 d2                	xor    %edx,%edx
  404513:	48 89 05 fe 6b 21 00 	mov    %rax,0x216bfe(%rip)        # 61b118 <stderr@@GLIBC_2.2.5+0xac8>
  40451a:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40451f:	83 fa 05             	cmp    $0x5,%edx
  404522:	0f 87 5d e4 ff ff    	ja     402985 <__sprintf_chk@plt+0xf5>
  404528:	89 d0                	mov    %edx,%eax
  40452a:	ff 24 c5 c8 2b 41 00 	jmpq   *0x412bc8(,%rax,8)
  404531:	bf e4 38 41 00       	mov    $0x4138e4,%edi
  404536:	e8 85 dc ff ff       	callq  4021c0 <getenv@plt>
  40453b:	48 85 c0             	test   %rax,%rax
  40453e:	0f 85 6e f4 ff ff    	jne    4039b2 <__sprintf_chk@plt+0x1122>
  404544:	e9 83 f4 ff ff       	jmpq   4039cc <__sprintf_chk@plt+0x113c>
  404549:	48 c7 05 84 5e 21 00 	movq   $0x413966,0x215e84(%rip)        # 61a3d8 <_fini@@Base+0x2084dc>
  404550:	66 39 41 00 
  404554:	48 c7 05 71 5e 21 00 	movq   $0x413966,0x215e71(%rip)        # 61a3d0 <_fini@@Base+0x2084d4>
  40455b:	66 39 41 00 
  40455f:	e9 20 ff ff ff       	jmpq   404484 <__sprintf_chk@plt+0x1bf4>
  404564:	48 89 c2             	mov    %rax,%rdx
  404567:	4c 89 e6             	mov    %r12,%rsi
  40456a:	bf 1d 39 41 00       	mov    $0x41391d,%edi
  40456f:	e8 0c 5a 00 00       	callq  409f80 <__sprintf_chk@plt+0x76f0>
  404574:	48 8b 1d d5 60 21 00 	mov    0x2160d5(%rip),%rbx        # 61a650 <stderr@@GLIBC_2.2.5>
  40457b:	ba 05 00 00 00       	mov    $0x5,%edx
  404580:	be 28 39 41 00       	mov    $0x413928,%esi
  404585:	31 ff                	xor    %edi,%edi
  404587:	e8 d4 dd ff ff       	callq  402360 <dcgettext@plt>
  40458c:	48 89 de             	mov    %rbx,%rsi
  40458f:	48 89 c7             	mov    %rax,%rdi
  404592:	bb 00 37 41 00       	mov    $0x413700,%ebx
  404597:	e8 84 df ff ff       	callq  402520 <fputs_unlocked@plt>
  40459c:	48 8b 0b             	mov    (%rbx),%rcx
  40459f:	48 85 c9             	test   %rcx,%rcx
  4045a2:	0f 84 2e ec ff ff    	je     4031d6 <__sprintf_chk@plt+0x946>
  4045a8:	48 8b 3d a1 60 21 00 	mov    0x2160a1(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  4045af:	ba 3e 39 41 00       	mov    $0x41393e,%edx
  4045b4:	be 01 00 00 00       	mov    $0x1,%esi
  4045b9:	31 c0                	xor    %eax,%eax
  4045bb:	48 83 c3 08          	add    $0x8,%rbx
  4045bf:	e8 4c e2 ff ff       	callq  402810 <__fprintf_chk@plt>
  4045c4:	eb d6                	jmp    40459c <__sprintf_chk@plt+0x1d0c>
  4045c6:	bf f5 38 41 00       	mov    $0x4138f5,%edi
  4045cb:	e8 f0 db ff ff       	callq  4021c0 <getenv@plt>
  4045d0:	49 89 c4             	mov    %rax,%r12
  4045d3:	48 85 c0             	test   %rax,%rax
  4045d6:	b8 27 38 41 00       	mov    $0x413827,%eax
  4045db:	4c 0f 44 e0          	cmove  %rax,%r12
  4045df:	e9 32 f2 ff ff       	jmpq   403816 <__sprintf_chk@plt+0xf86>
  4045e4:	be 0a 00 00 00       	mov    $0xa,%esi
  4045e9:	e8 12 de ff ff       	callq  402400 <__overflow@plt>
  4045ee:	e9 4b f0 ff ff       	jmpq   40363e <__sprintf_chk@plt+0xdae>
  4045f3:	48 c7 05 d2 5d 21 00 	movq   $0x413975,0x215dd2(%rip)        # 61a3d0 <_fini@@Base+0x2084d4>
  4045fa:	75 39 41 00 
  4045fe:	48 c7 05 cf 5d 21 00 	movq   $0x413969,0x215dcf(%rip)        # 61a3d8 <_fini@@Base+0x2084dc>
  404605:	69 39 41 00 
  404609:	e9 9a f2 ff ff       	jmpq   4038a8 <__sprintf_chk@plt+0x1018>
  40460e:	48 8b 35 bb 5d 21 00 	mov    0x215dbb(%rip),%rsi        # 61a3d0 <_fini@@Base+0x2084d4>
  404615:	ba 02 00 00 00       	mov    $0x2,%edx
  40461a:	31 ff                	xor    %edi,%edi
  40461c:	e8 3f dd ff ff       	callq  402360 <dcgettext@plt>
  404621:	48 8b 35 b0 5d 21 00 	mov    0x215db0(%rip),%rsi        # 61a3d8 <_fini@@Base+0x2084dc>
  404628:	ba 02 00 00 00       	mov    $0x2,%edx
  40462d:	31 ff                	xor    %edi,%edi
  40462f:	48 89 05 9a 5d 21 00 	mov    %rax,0x215d9a(%rip)        # 61a3d0 <_fini@@Base+0x2084d4>
  404636:	e8 25 dd ff ff       	callq  402360 <dcgettext@plt>
  40463b:	48 89 05 96 5d 21 00 	mov    %rax,0x215d96(%rip)        # 61a3d8 <_fini@@Base+0x2084dc>
  404642:	e9 61 f2 ff ff       	jmpq   4038a8 <__sprintf_chk@plt+0x1018>
  404647:	ba 05 00 00 00       	mov    $0x5,%edx
  40464c:	be 98 5d 41 00       	mov    $0x415d98,%esi
  404651:	31 ff                	xor    %edi,%edi
  404653:	e8 08 dd ff ff       	callq  402360 <dcgettext@plt>
  404658:	31 f6                	xor    %esi,%esi
  40465a:	48 89 c2             	mov    %rax,%rdx
  40465d:	31 ff                	xor    %edi,%edi
  40465f:	31 c0                	xor    %eax,%eax
  404661:	e8 0a e1 ff ff       	callq  402770 <error@plt>
  404666:	48 8b 3d ab 6a 21 00 	mov    0x216aab(%rip),%rdi        # 61b118 <stderr@@GLIBC_2.2.5+0xac8>
  40466d:	e8 7e db ff ff       	callq  4021f0 <free@plt>
  404672:	48 8b 3d a7 6a 21 00 	mov    0x216aa7(%rip),%rdi        # 61b120 <stderr@@GLIBC_2.2.5+0xad0>
  404679:	48 85 ff             	test   %rdi,%rdi
  40467c:	0f 84 05 01 00 00    	je     404787 <__sprintf_chk@plt+0x1ef7>
  404682:	4c 8b 6f 20          	mov    0x20(%rdi),%r13
  404686:	e8 65 db ff ff       	callq  4021f0 <free@plt>
  40468b:	4c 89 ef             	mov    %r13,%rdi
  40468e:	eb e9                	jmp    404679 <__sprintf_chk@plt+0x1de9>
  404690:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  404695:	48 8d 50 01          	lea    0x1(%rax),%rdx
  404699:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
  40469e:	80 38 3d             	cmpb   $0x3d,(%rax)
  4046a1:	ba 05 00 00 00       	mov    $0x5,%edx
  4046a6:	0f 85 7c fe ff ff    	jne    404528 <__sprintf_chk@plt+0x1c98>
  4046ac:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  4046b1:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4046b6:	49 8d 4d 10          	lea    0x10(%r13),%rcx
  4046ba:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
  4046bf:	30 d2                	xor    %dl,%dl
  4046c1:	49 89 45 18          	mov    %rax,0x18(%r13)
  4046c5:	e8 16 03 00 00       	callq  4049e0 <__sprintf_chk@plt+0x2150>
  4046ca:	3c 01                	cmp    $0x1,%al
  4046cc:	19 d2                	sbb    %edx,%edx
  4046ce:	83 e2 05             	and    $0x5,%edx
  4046d1:	e9 49 fe ff ff       	jmpq   40451f <__sprintf_chk@plt+0x1c8f>
  4046d6:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  4046db:	ba 05 00 00 00       	mov    $0x5,%edx
  4046e0:	80 38 00             	cmpb   $0x0,(%rax)
  4046e3:	0f 84 3f fe ff ff    	je     404528 <__sprintf_chk@plt+0x1c98>
  4046e9:	48 8d 50 01          	lea    0x1(%rax),%rdx
  4046ed:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
  4046f2:	0f b6 00             	movzbl (%rax),%eax
  4046f5:	ba 02 00 00 00       	mov    $0x2,%edx
  4046fa:	88 44 24 31          	mov    %al,0x31(%rsp)
  4046fe:	e9 25 fe ff ff       	jmpq   404528 <__sprintf_chk@plt+0x1c98>
  404703:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  404708:	0f b6 08             	movzbl (%rax),%ecx
  40470b:	80 f9 2a             	cmp    $0x2a,%cl
  40470e:	0f 84 c0 00 00 00    	je     4047d4 <__sprintf_chk@plt+0x1f44>
  404714:	80 f9 3a             	cmp    $0x3a,%cl
  404717:	0f 84 a9 00 00 00    	je     4047c6 <__sprintf_chk@plt+0x1f36>
  40471d:	84 c9                	test   %cl,%cl
  40471f:	74 6d                	je     40478e <__sprintf_chk@plt+0x1efe>
  404721:	48 8d 50 01          	lea    0x1(%rax),%rdx
  404725:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
  40472a:	0f b6 00             	movzbl (%rax),%eax
  40472d:	ba 01 00 00 00       	mov    $0x1,%edx
  404732:	88 44 24 30          	mov    %al,0x30(%rsp)
  404736:	e9 ed fd ff ff       	jmpq   404528 <__sprintf_chk@plt+0x1c98>
  40473b:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
  404740:	45 31 ff             	xor    %r15d,%r15d
  404743:	48 8d 50 01          	lea    0x1(%rax),%rdx
  404747:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
  40474c:	80 38 3d             	cmpb   $0x3d,(%rax)
  40474f:	ba 05 00 00 00       	mov    $0x5,%edx
  404754:	0f 85 ce fd ff ff    	jne    404528 <__sprintf_chk@plt+0x1c98>
  40475a:	eb 16                	jmp    404772 <__sprintf_chk@plt+0x1ee2>
  40475c:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  404761:	49 83 c7 01          	add    $0x1,%r15
  404765:	e8 e6 dd ff ff       	callq  402550 <strcmp@plt>
  40476a:	85 c0                	test   %eax,%eax
  40476c:	0f 84 b6 00 00 00    	je     404828 <__sprintf_chk@plt+0x1f98>
  404772:	4a 8b 34 fd e0 35 41 	mov    0x4135e0(,%r15,8),%rsi
  404779:	00 
  40477a:	4d 63 f7             	movslq %r15d,%r14
  40477d:	48 85 f6             	test   %rsi,%rsi
  404780:	75 da                	jne    40475c <__sprintf_chk@plt+0x1ecc>
  404782:	e9 d0 00 00 00       	jmpq   404857 <__sprintf_chk@plt+0x1fc7>
  404787:	c6 05 9b 69 21 00 00 	movb   $0x0,0x21699b(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  40478e:	48 83 3d ba 5c 21 00 	cmpq   $0x6,0x215cba(%rip)        # 61a450 <_fini@@Base+0x208554>
  404795:	06 
  404796:	0f 85 46 f3 ff ff    	jne    403ae2 <__sprintf_chk@plt+0x1252>
  40479c:	48 8b 3d b5 5c 21 00 	mov    0x215cb5(%rip),%rdi        # 61a458 <_fini@@Base+0x20855c>
  4047a3:	ba 06 00 00 00       	mov    $0x6,%edx
  4047a8:	be 89 39 41 00       	mov    $0x413989,%esi
  4047ad:	e8 8e da ff ff       	callq  402240 <strncmp@plt>
  4047b2:	85 c0                	test   %eax,%eax
  4047b4:	0f 85 28 f3 ff ff    	jne    403ae2 <__sprintf_chk@plt+0x1252>
  4047ba:	c6 05 d7 69 21 00 01 	movb   $0x1,0x2169d7(%rip)        # 61b198 <stderr@@GLIBC_2.2.5+0xb48>
  4047c1:	e9 1c f3 ff ff       	jmpq   403ae2 <__sprintf_chk@plt+0x1252>
  4047c6:	48 83 c0 01          	add    $0x1,%rax
  4047ca:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  4047cf:	e9 4b fd ff ff       	jmpq   40451f <__sprintf_chk@plt+0x1c8f>
  4047d4:	bf 28 00 00 00       	mov    $0x28,%edi
  4047d9:	e8 62 c4 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  4047de:	49 89 c5             	mov    %rax,%r13
  4047e1:	48 8b 05 38 69 21 00 	mov    0x216938(%rip),%rax        # 61b120 <stderr@@GLIBC_2.2.5+0xad0>
  4047e8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  4047ed:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
  4047f2:	ba 01 00 00 00       	mov    $0x1,%edx
  4047f7:	4c 89 e9             	mov    %r13,%rcx
  4047fa:	48 83 44 24 38 01    	addq   $0x1,0x38(%rsp)
  404800:	4c 89 2d 19 69 21 00 	mov    %r13,0x216919(%rip)        # 61b120 <stderr@@GLIBC_2.2.5+0xad0>
  404807:	49 89 45 20          	mov    %rax,0x20(%r13)
  40480b:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  404810:	49 89 45 08          	mov    %rax,0x8(%r13)
  404814:	e8 c7 01 00 00       	callq  4049e0 <__sprintf_chk@plt+0x2150>
  404819:	3c 01                	cmp    $0x1,%al
  40481b:	19 d2                	sbb    %edx,%edx
  40481d:	83 e2 02             	and    $0x2,%edx
  404820:	83 c2 03             	add    $0x3,%edx
  404823:	e9 f7 fc ff ff       	jmpq   40451f <__sprintf_chk@plt+0x1c8f>
  404828:	49 c1 e6 04          	shl    $0x4,%r14
  40482c:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
  404831:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  404836:	49 8d 8e e0 a3 61 00 	lea    0x61a3e0(%r14),%rcx
  40483d:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
  404842:	31 d2                	xor    %edx,%edx
  404844:	48 89 41 08          	mov    %rax,0x8(%rcx)
  404848:	e8 93 01 00 00       	callq  4049e0 <__sprintf_chk@plt+0x2150>
  40484d:	31 d2                	xor    %edx,%edx
  40484f:	84 c0                	test   %al,%al
  404851:	0f 85 d1 fc ff ff    	jne    404528 <__sprintf_chk@plt+0x1c98>
  404857:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  40485c:	e8 cf a0 00 00       	callq  40e930 <__sprintf_chk@plt+0xc0a0>
  404861:	ba 05 00 00 00       	mov    $0x5,%edx
  404866:	49 89 c6             	mov    %rax,%r14
  404869:	be fb 39 41 00       	mov    $0x4139fb,%esi
  40486e:	31 ff                	xor    %edi,%edi
  404870:	e8 eb da ff ff       	callq  402360 <dcgettext@plt>
  404875:	4c 89 f1             	mov    %r14,%rcx
  404878:	48 89 c2             	mov    %rax,%rdx
  40487b:	31 f6                	xor    %esi,%esi
  40487d:	31 ff                	xor    %edi,%edi
  40487f:	31 c0                	xor    %eax,%eax
  404881:	e8 ea de ff ff       	callq  402770 <error@plt>
  404886:	ba 05 00 00 00       	mov    $0x5,%edx
  40488b:	e9 98 fc ff ff       	jmpq   404528 <__sprintf_chk@plt+0x1c98>
  404890:	31 ed                	xor    %ebp,%ebp
  404892:	49 89 d1             	mov    %rdx,%r9
  404895:	5e                   	pop    %rsi
  404896:	48 89 e2             	mov    %rsp,%rdx
  404899:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  40489d:	50                   	push   %rax
  40489e:	54                   	push   %rsp
  40489f:	49 c7 c0 d0 1e 41 00 	mov    $0x411ed0,%r8
  4048a6:	48 c7 c1 60 1e 41 00 	mov    $0x411e60,%rcx
  4048ad:	48 c7 c7 c0 28 40 00 	mov    $0x4028c0,%rdi
  4048b4:	e8 37 dc ff ff       	callq  4024f0 <__libc_start_main@plt>
  4048b9:	f4                   	hlt    
  4048ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4048c0:	b8 ff a5 61 00       	mov    $0x61a5ff,%eax
  4048c5:	55                   	push   %rbp
  4048c6:	48 2d f8 a5 61 00    	sub    $0x61a5f8,%rax
  4048cc:	48 83 f8 0e          	cmp    $0xe,%rax
  4048d0:	48 89 e5             	mov    %rsp,%rbp
  4048d3:	77 02                	ja     4048d7 <__sprintf_chk@plt+0x2047>
  4048d5:	5d                   	pop    %rbp
  4048d6:	c3                   	retq   
  4048d7:	b8 00 00 00 00       	mov    $0x0,%eax
  4048dc:	48 85 c0             	test   %rax,%rax
  4048df:	74 f4                	je     4048d5 <__sprintf_chk@plt+0x2045>
  4048e1:	5d                   	pop    %rbp
  4048e2:	bf f8 a5 61 00       	mov    $0x61a5f8,%edi
  4048e7:	ff e0                	jmpq   *%rax
  4048e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4048f0:	b8 f8 a5 61 00       	mov    $0x61a5f8,%eax
  4048f5:	55                   	push   %rbp
  4048f6:	48 2d f8 a5 61 00    	sub    $0x61a5f8,%rax
  4048fc:	48 c1 f8 03          	sar    $0x3,%rax
  404900:	48 89 e5             	mov    %rsp,%rbp
  404903:	48 89 c2             	mov    %rax,%rdx
  404906:	48 c1 ea 3f          	shr    $0x3f,%rdx
  40490a:	48 01 d0             	add    %rdx,%rax
  40490d:	48 d1 f8             	sar    %rax
  404910:	75 02                	jne    404914 <__sprintf_chk@plt+0x2084>
  404912:	5d                   	pop    %rbp
  404913:	c3                   	retq   
  404914:	ba 00 00 00 00       	mov    $0x0,%edx
  404919:	48 85 d2             	test   %rdx,%rdx
  40491c:	74 f4                	je     404912 <__sprintf_chk@plt+0x2082>
  40491e:	5d                   	pop    %rbp
  40491f:	48 89 c6             	mov    %rax,%rsi
  404922:	bf f8 a5 61 00       	mov    $0x61a5f8,%edi
  404927:	ff e2                	jmpq   *%rdx
  404929:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  404930:	80 3d 21 5d 21 00 00 	cmpb   $0x0,0x215d21(%rip)        # 61a658 <stderr@@GLIBC_2.2.5+0x8>
  404937:	75 11                	jne    40494a <__sprintf_chk@plt+0x20ba>
  404939:	55                   	push   %rbp
  40493a:	48 89 e5             	mov    %rsp,%rbp
  40493d:	e8 7e ff ff ff       	callq  4048c0 <__sprintf_chk@plt+0x2030>
  404942:	5d                   	pop    %rbp
  404943:	c6 05 0e 5d 21 00 01 	movb   $0x1,0x215d0e(%rip)        # 61a658 <stderr@@GLIBC_2.2.5+0x8>
  40494a:	f3 c3                	repz retq 
  40494c:	0f 1f 40 00          	nopl   0x0(%rax)
  404950:	48 83 3d a8 54 21 00 	cmpq   $0x0,0x2154a8(%rip)        # 619e00 <_fini@@Base+0x207f04>
  404957:	00 
  404958:	74 1e                	je     404978 <__sprintf_chk@plt+0x20e8>
  40495a:	b8 00 00 00 00       	mov    $0x0,%eax
  40495f:	48 85 c0             	test   %rax,%rax
  404962:	74 14                	je     404978 <__sprintf_chk@plt+0x20e8>
  404964:	55                   	push   %rbp
  404965:	bf 00 9e 61 00       	mov    $0x619e00,%edi
  40496a:	48 89 e5             	mov    %rsp,%rbp
  40496d:	ff d0                	callq  *%rax
  40496f:	5d                   	pop    %rbp
  404970:	e9 7b ff ff ff       	jmpq   4048f0 <__sprintf_chk@plt+0x2060>
  404975:	0f 1f 00             	nopl   (%rax)
  404978:	e9 73 ff ff ff       	jmpq   4048f0 <__sprintf_chk@plt+0x2060>
  40497d:	0f 1f 00             	nopl   (%rax)
  404980:	48 8b 07             	mov    (%rdi),%rax
  404983:	31 d2                	xor    %edx,%edx
  404985:	48 f7 f6             	div    %rsi
  404988:	48 89 d0             	mov    %rdx,%rax
  40498b:	c3                   	retq   
  40498c:	0f 1f 40 00          	nopl   0x0(%rax)
  404990:	31 c0                	xor    %eax,%eax
  404992:	48 8b 16             	mov    (%rsi),%rdx
  404995:	48 39 17             	cmp    %rdx,(%rdi)
  404998:	74 06                	je     4049a0 <__sprintf_chk@plt+0x2110>
  40499a:	f3 c3                	repz retq 
  40499c:	0f 1f 40 00          	nopl   0x0(%rax)
  4049a0:	48 8b 46 08          	mov    0x8(%rsi),%rax
  4049a4:	48 39 47 08          	cmp    %rax,0x8(%rdi)
  4049a8:	0f 94 c0             	sete   %al
  4049ab:	c3                   	retq   
  4049ac:	0f 1f 40 00          	nopl   0x0(%rax)
  4049b0:	8b 05 82 66 21 00    	mov    0x216682(%rip),%eax        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  4049b6:	85 c0                	test   %eax,%eax
  4049b8:	75 06                	jne    4049c0 <__sprintf_chk@plt+0x2130>
  4049ba:	89 3d 78 66 21 00    	mov    %edi,0x216678(%rip)        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  4049c0:	f3 c3                	repz retq 
  4049c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4049c9:	1f 84 00 00 00 00 00 
  4049d0:	e9 1b d8 ff ff       	jmpq   4021f0 <free@plt>
  4049d5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4049dc:	00 00 00 00 
  4049e0:	41 56                	push   %r14
  4049e2:	4c 8b 06             	mov    (%rsi),%r8
  4049e5:	31 c0                	xor    %eax,%eax
  4049e7:	4c 8b 0f             	mov    (%rdi),%r9
  4049ea:	45 31 d2             	xor    %r10d,%r10d
  4049ed:	45 31 db             	xor    %r11d,%r11d
  4049f0:	41 54                	push   %r12
  4049f2:	49 bc 00 00 00 00 00 	movabs $0x7e000000000000,%r12
  4049f9:	00 7e 00 
  4049fc:	55                   	push   %rbp
  4049fd:	48 89 cd             	mov    %rcx,%rbp
  404a00:	53                   	push   %rbx
  404a01:	bb 01 00 00 00       	mov    $0x1,%ebx
  404a06:	83 f8 02             	cmp    $0x2,%eax
  404a09:	74 4e                	je     404a59 <__sprintf_chk@plt+0x21c9>
  404a0b:	0f 86 0f 01 00 00    	jbe    404b20 <__sprintf_chk@plt+0x2290>
  404a11:	83 f8 03             	cmp    $0x3,%eax
  404a14:	0f 84 c6 00 00 00    	je     404ae0 <__sprintf_chk@plt+0x2250>
  404a1a:	83 f8 04             	cmp    $0x4,%eax
  404a1d:	0f 1f 00             	nopl   (%rax)
  404a20:	75 4e                	jne    404a70 <__sprintf_chk@plt+0x21e0>
  404a22:	41 0f b6 00          	movzbl (%r8),%eax
  404a26:	8d 48 c0             	lea    -0x40(%rax),%ecx
  404a29:	80 f9 3e             	cmp    $0x3e,%cl
  404a2c:	76 72                	jbe    404aa0 <__sprintf_chk@plt+0x2210>
  404a2e:	3c 3f                	cmp    $0x3f,%al
  404a30:	0f 84 7a 02 00 00    	je     404cb0 <__sprintf_chk@plt+0x2420>
  404a36:	31 c0                	xor    %eax,%eax
  404a38:	4c 89 0f             	mov    %r9,(%rdi)
  404a3b:	4c 89 06             	mov    %r8,(%rsi)
  404a3e:	4c 89 55 00          	mov    %r10,0x0(%rbp)
  404a42:	5b                   	pop    %rbx
  404a43:	5d                   	pop    %rbp
  404a44:	41 5c                	pop    %r12
  404a46:	41 5e                	pop    %r14
  404a48:	c3                   	retq   
  404a49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  404a50:	46 8d 5c d8 d0       	lea    -0x30(%rax,%r11,8),%r11d
  404a55:	49 83 c0 01          	add    $0x1,%r8
  404a59:	41 0f b6 00          	movzbl (%r8),%eax
  404a5d:	8d 48 d0             	lea    -0x30(%rax),%ecx
  404a60:	80 f9 07             	cmp    $0x7,%cl
  404a63:	76 eb                	jbe    404a50 <__sprintf_chk@plt+0x21c0>
  404a65:	45 88 19             	mov    %r11b,(%r9)
  404a68:	49 83 c2 01          	add    $0x1,%r10
  404a6c:	49 83 c1 01          	add    $0x1,%r9
  404a70:	41 0f b6 00          	movzbl (%r8),%eax
  404a74:	3c 3d                	cmp    $0x3d,%al
  404a76:	74 42                	je     404aba <__sprintf_chk@plt+0x222a>
  404a78:	0f 8e c2 00 00 00    	jle    404b40 <__sprintf_chk@plt+0x22b0>
  404a7e:	3c 5c                	cmp    $0x5c,%al
  404a80:	0f 84 1a 02 00 00    	je     404ca0 <__sprintf_chk@plt+0x2410>
  404a86:	3c 5e                	cmp    $0x5e,%al
  404a88:	0f 85 d2 00 00 00    	jne    404b60 <__sprintf_chk@plt+0x22d0>
  404a8e:	49 83 c0 01          	add    $0x1,%r8
  404a92:	41 0f b6 00          	movzbl (%r8),%eax
  404a96:	8d 48 c0             	lea    -0x40(%rax),%ecx
  404a99:	80 f9 3e             	cmp    $0x3e,%cl
  404a9c:	77 90                	ja     404a2e <__sprintf_chk@plt+0x219e>
  404a9e:	66 90                	xchg   %ax,%ax
  404aa0:	83 e0 1f             	and    $0x1f,%eax
  404aa3:	49 83 c0 01          	add    $0x1,%r8
  404aa7:	49 83 c2 01          	add    $0x1,%r10
  404aab:	41 88 01             	mov    %al,(%r9)
  404aae:	41 0f b6 00          	movzbl (%r8),%eax
  404ab2:	49 83 c1 01          	add    $0x1,%r9
  404ab6:	3c 3d                	cmp    $0x3d,%al
  404ab8:	75 be                	jne    404a78 <__sprintf_chk@plt+0x21e8>
  404aba:	84 d2                	test   %dl,%dl
  404abc:	0f 84 9e 00 00 00    	je     404b60 <__sprintf_chk@plt+0x22d0>
  404ac2:	b8 01 00 00 00       	mov    $0x1,%eax
  404ac7:	e9 6c ff ff ff       	jmpq   404a38 <__sprintf_chk@plt+0x21a8>
  404acc:	0f 1f 40 00          	nopl   0x0(%rax)
  404ad0:	41 c1 e3 04          	shl    $0x4,%r11d
  404ad4:	49 83 c0 01          	add    $0x1,%r8
  404ad8:	46 8d 5c 18 a9       	lea    -0x57(%rax,%r11,1),%r11d
  404add:	0f 1f 00             	nopl   (%rax)
  404ae0:	41 0f b6 00          	movzbl (%r8),%eax
  404ae4:	8d 48 d0             	lea    -0x30(%rax),%ecx
  404ae7:	80 f9 36             	cmp    $0x36,%cl
  404aea:	0f 87 75 ff ff ff    	ja     404a65 <__sprintf_chk@plt+0x21d5>
  404af0:	49 89 de             	mov    %rbx,%r14
  404af3:	49 d3 e6             	shl    %cl,%r14
  404af6:	41 f7 c6 00 00 7e 00 	test   $0x7e0000,%r14d
  404afd:	75 79                	jne    404b78 <__sprintf_chk@plt+0x22e8>
  404aff:	4d 85 e6             	test   %r12,%r14
  404b02:	75 cc                	jne    404ad0 <__sprintf_chk@plt+0x2240>
  404b04:	41 f7 c6 ff 03 00 00 	test   $0x3ff,%r14d
  404b0b:	0f 84 54 ff ff ff    	je     404a65 <__sprintf_chk@plt+0x21d5>
  404b11:	41 c1 e3 04          	shl    $0x4,%r11d
  404b15:	49 83 c0 01          	add    $0x1,%r8
  404b19:	46 8d 5c 18 d0       	lea    -0x30(%rax,%r11,1),%r11d
  404b1e:	eb c0                	jmp    404ae0 <__sprintf_chk@plt+0x2250>
  404b20:	83 f8 01             	cmp    $0x1,%eax
  404b23:	0f 85 47 ff ff ff    	jne    404a70 <__sprintf_chk@plt+0x21e0>
  404b29:	41 0f b6 00          	movzbl (%r8),%eax
  404b2d:	3c 78                	cmp    $0x78,%al
  404b2f:	0f 87 5b 01 00 00    	ja     404c90 <__sprintf_chk@plt+0x2400>
  404b35:	0f b6 c8             	movzbl %al,%ecx
  404b38:	ff 24 cd 40 1f 41 00 	jmpq   *0x411f40(,%rcx,8)
  404b3f:	90                   	nop
  404b40:	84 c0                	test   %al,%al
  404b42:	74 04                	je     404b48 <__sprintf_chk@plt+0x22b8>
  404b44:	3c 3a                	cmp    $0x3a,%al
  404b46:	75 18                	jne    404b60 <__sprintf_chk@plt+0x22d0>
  404b48:	b8 05 00 00 00       	mov    $0x5,%eax
  404b4d:	83 f8 06             	cmp    $0x6,%eax
  404b50:	0f 95 c0             	setne  %al
  404b53:	e9 e0 fe ff ff       	jmpq   404a38 <__sprintf_chk@plt+0x21a8>
  404b58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404b5f:	00 
  404b60:	41 88 01             	mov    %al,(%r9)
  404b63:	49 83 c0 01          	add    $0x1,%r8
  404b67:	49 83 c2 01          	add    $0x1,%r10
  404b6b:	49 83 c1 01          	add    $0x1,%r9
  404b6f:	e9 fc fe ff ff       	jmpq   404a70 <__sprintf_chk@plt+0x21e0>
  404b74:	0f 1f 40 00          	nopl   0x0(%rax)
  404b78:	49 83 c0 01          	add    $0x1,%r8
  404b7c:	41 c1 e3 04          	shl    $0x4,%r11d
  404b80:	46 8d 5c 18 c9       	lea    -0x37(%rax,%r11,1),%r11d
  404b85:	41 0f b6 00          	movzbl (%r8),%eax
  404b89:	8d 48 d0             	lea    -0x30(%rax),%ecx
  404b8c:	80 f9 36             	cmp    $0x36,%cl
  404b8f:	0f 87 d0 fe ff ff    	ja     404a65 <__sprintf_chk@plt+0x21d5>
  404b95:	e9 56 ff ff ff       	jmpq   404af0 <__sprintf_chk@plt+0x2260>
  404b9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  404ba0:	b8 03 00 00 00       	mov    $0x3,%eax
  404ba5:	45 31 db             	xor    %r11d,%r11d
  404ba8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404baf:	00 
  404bb0:	49 83 c0 01          	add    $0x1,%r8
  404bb4:	83 f8 04             	cmp    $0x4,%eax
  404bb7:	0f 86 49 fe ff ff    	jbe    404a06 <__sprintf_chk@plt+0x2176>
  404bbd:	eb 8e                	jmp    404b4d <__sprintf_chk@plt+0x22bd>
  404bbf:	90                   	nop
  404bc0:	41 bb 1b 00 00 00    	mov    $0x1b,%r11d
  404bc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  404bcd:	00 00 00 
  404bd0:	45 88 19             	mov    %r11b,(%r9)
  404bd3:	49 83 c2 01          	add    $0x1,%r10
  404bd7:	49 83 c1 01          	add    $0x1,%r9
  404bdb:	31 c0                	xor    %eax,%eax
  404bdd:	eb d1                	jmp    404bb0 <__sprintf_chk@plt+0x2320>
  404bdf:	90                   	nop
  404be0:	41 bb 20 00 00 00    	mov    $0x20,%r11d
  404be6:	eb e8                	jmp    404bd0 <__sprintf_chk@plt+0x2340>
  404be8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404bef:	00 
  404bf0:	b8 06 00 00 00       	mov    $0x6,%eax
  404bf5:	eb b9                	jmp    404bb0 <__sprintf_chk@plt+0x2320>
  404bf7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  404bfe:	00 00 
  404c00:	44 8d 58 d0          	lea    -0x30(%rax),%r11d
  404c04:	b8 02 00 00 00       	mov    $0x2,%eax
  404c09:	eb a5                	jmp    404bb0 <__sprintf_chk@plt+0x2320>
  404c0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404c10:	41 bb 7f 00 00 00    	mov    $0x7f,%r11d
  404c16:	eb b8                	jmp    404bd0 <__sprintf_chk@plt+0x2340>
  404c18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404c1f:	00 
  404c20:	41 bb 07 00 00 00    	mov    $0x7,%r11d
  404c26:	eb a8                	jmp    404bd0 <__sprintf_chk@plt+0x2340>
  404c28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404c2f:	00 
  404c30:	41 bb 08 00 00 00    	mov    $0x8,%r11d
  404c36:	eb 98                	jmp    404bd0 <__sprintf_chk@plt+0x2340>
  404c38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404c3f:	00 
  404c40:	41 bb 09 00 00 00    	mov    $0x9,%r11d
  404c46:	eb 88                	jmp    404bd0 <__sprintf_chk@plt+0x2340>
  404c48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404c4f:	00 
  404c50:	41 bb 0b 00 00 00    	mov    $0xb,%r11d
  404c56:	e9 75 ff ff ff       	jmpq   404bd0 <__sprintf_chk@plt+0x2340>
  404c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404c60:	41 bb 0a 00 00 00    	mov    $0xa,%r11d
  404c66:	e9 65 ff ff ff       	jmpq   404bd0 <__sprintf_chk@plt+0x2340>
  404c6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404c70:	41 bb 0d 00 00 00    	mov    $0xd,%r11d
  404c76:	e9 55 ff ff ff       	jmpq   404bd0 <__sprintf_chk@plt+0x2340>
  404c7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404c80:	41 bb 0c 00 00 00    	mov    $0xc,%r11d
  404c86:	e9 45 ff ff ff       	jmpq   404bd0 <__sprintf_chk@plt+0x2340>
  404c8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404c90:	41 89 c3             	mov    %eax,%r11d
  404c93:	e9 38 ff ff ff       	jmpq   404bd0 <__sprintf_chk@plt+0x2340>
  404c98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  404c9f:	00 
  404ca0:	49 83 c0 01          	add    $0x1,%r8
  404ca4:	e9 80 fe ff ff       	jmpq   404b29 <__sprintf_chk@plt+0x2299>
  404ca9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  404cb0:	41 c6 01 7f          	movb   $0x7f,(%r9)
  404cb4:	49 83 c2 01          	add    $0x1,%r10
  404cb8:	49 83 c1 01          	add    $0x1,%r9
  404cbc:	e9 af fd ff ff       	jmpq   404a70 <__sprintf_chk@plt+0x21e0>
  404cc1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  404cc8:	0f 1f 84 00 00 00 00 
  404ccf:	00 
  404cd0:	89 ff                	mov    %edi,%edi
  404cd2:	31 c0                	xor    %eax,%eax
  404cd4:	48 c1 e7 04          	shl    $0x4,%rdi
  404cd8:	48 8b 97 e0 a3 61 00 	mov    0x61a3e0(%rdi),%rdx
  404cdf:	48 8b b7 e8 a3 61 00 	mov    0x61a3e8(%rdi),%rsi
  404ce6:	48 85 d2             	test   %rdx,%rdx
  404ce9:	74 11                	je     404cfc <__sprintf_chk@plt+0x246c>
  404ceb:	48 83 fa 01          	cmp    $0x1,%rdx
  404cef:	74 1f                	je     404d10 <__sprintf_chk@plt+0x2480>
  404cf1:	48 83 fa 02          	cmp    $0x2,%rdx
  404cf5:	b8 01 00 00 00       	mov    $0x1,%eax
  404cfa:	74 04                	je     404d00 <__sprintf_chk@plt+0x2470>
  404cfc:	f3 c3                	repz retq 
  404cfe:	66 90                	xchg   %ax,%ax
  404d00:	bf 33 37 41 00       	mov    $0x413733,%edi
  404d05:	b9 02 00 00 00       	mov    $0x2,%ecx
  404d0a:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  404d0c:	0f 95 c0             	setne  %al
  404d0f:	c3                   	retq   
  404d10:	80 3e 30             	cmpb   $0x30,(%rsi)
  404d13:	0f 95 c0             	setne  %al
  404d16:	c3                   	retq   
  404d17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  404d1e:	00 00 
  404d20:	41 55                	push   %r13
  404d22:	41 89 d5             	mov    %edx,%r13d
  404d25:	41 54                	push   %r12
  404d27:	49 89 f4             	mov    %rsi,%r12
  404d2a:	55                   	push   %rbp
  404d2b:	48 89 fd             	mov    %rdi,%rbp
  404d2e:	bf 20 00 00 00       	mov    $0x20,%edi
  404d33:	53                   	push   %rbx
  404d34:	48 83 ec 08          	sub    $0x8,%rsp
  404d38:	e8 03 bf 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  404d3d:	48 89 c3             	mov    %rax,%rbx
  404d40:	31 c0                	xor    %eax,%eax
  404d42:	4d 85 e4             	test   %r12,%r12
  404d45:	74 08                	je     404d4f <__sprintf_chk@plt+0x24bf>
  404d47:	4c 89 e7             	mov    %r12,%rdi
  404d4a:	e8 e1 c0 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  404d4f:	48 89 43 08          	mov    %rax,0x8(%rbx)
  404d53:	31 c0                	xor    %eax,%eax
  404d55:	48 85 ed             	test   %rbp,%rbp
  404d58:	74 08                	je     404d62 <__sprintf_chk@plt+0x24d2>
  404d5a:	48 89 ef             	mov    %rbp,%rdi
  404d5d:	e8 ce c0 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  404d62:	48 89 03             	mov    %rax,(%rbx)
  404d65:	48 8b 05 24 64 21 00 	mov    0x216424(%rip),%rax        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  404d6c:	44 88 6b 10          	mov    %r13b,0x10(%rbx)
  404d70:	48 89 1d 19 64 21 00 	mov    %rbx,0x216419(%rip)        # 61b190 <stderr@@GLIBC_2.2.5+0xb40>
  404d77:	48 89 43 18          	mov    %rax,0x18(%rbx)
  404d7b:	48 83 c4 08          	add    $0x8,%rsp
  404d7f:	5b                   	pop    %rbx
  404d80:	5d                   	pop    %rbp
  404d81:	41 5c                	pop    %r12
  404d83:	41 5d                	pop    %r13
  404d85:	c3                   	retq   
  404d86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  404d8d:	00 00 00 
  404d90:	53                   	push   %rbx
  404d91:	48 89 fb             	mov    %rdi,%rbx
  404d94:	48 8b 3f             	mov    (%rdi),%rdi
  404d97:	e8 54 d4 ff ff       	callq  4021f0 <free@plt>
  404d9c:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  404da0:	e8 4b d4 ff ff       	callq  4021f0 <free@plt>
  404da5:	48 8b bb a8 00 00 00 	mov    0xa8(%rbx),%rdi
  404dac:	48 81 ff 6a a5 61 00 	cmp    $0x61a56a,%rdi
  404db3:	74 0b                	je     404dc0 <__sprintf_chk@plt+0x2530>
  404db5:	5b                   	pop    %rbx
  404db6:	e9 15 da ff ff       	jmpq   4027d0 <freecon@plt>
  404dbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404dc0:	5b                   	pop    %rbx
  404dc1:	c3                   	retq   
  404dc2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  404dc9:	1f 84 00 00 00 00 00 
  404dd0:	53                   	push   %rbx
  404dd1:	31 db                	xor    %ebx,%ebx
  404dd3:	48 83 3d d5 63 21 00 	cmpq   $0x0,0x2163d5(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404dda:	00 
  404ddb:	74 20                	je     404dfd <__sprintf_chk@plt+0x256d>
  404ddd:	0f 1f 00             	nopl   (%rax)
  404de0:	48 8b 05 c1 63 21 00 	mov    0x2163c1(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404de7:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
  404deb:	48 83 c3 01          	add    $0x1,%rbx
  404def:	e8 9c ff ff ff       	callq  404d90 <__sprintf_chk@plt+0x2500>
  404df4:	48 39 1d b5 63 21 00 	cmp    %rbx,0x2163b5(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404dfb:	77 e3                	ja     404de0 <__sprintf_chk@plt+0x2550>
  404dfd:	48 c7 05 a8 63 21 00 	movq   $0x0,0x2163a8(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404e04:	00 00 00 00 
  404e08:	c6 05 6d 63 21 00 00 	movb   $0x0,0x21636d(%rip)        # 61b17c <stderr@@GLIBC_2.2.5+0xb2c>
  404e0f:	c7 05 5f 63 21 00 00 	movl   $0x0,0x21635f(%rip)        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  404e16:	00 00 00 
  404e19:	c7 05 51 63 21 00 00 	movl   $0x0,0x216351(%rip)        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  404e20:	00 00 00 
  404e23:	c7 05 43 63 21 00 00 	movl   $0x0,0x216343(%rip)        # 61b170 <stderr@@GLIBC_2.2.5+0xb20>
  404e2a:	00 00 00 
  404e2d:	c7 05 31 63 21 00 00 	movl   $0x0,0x216331(%rip)        # 61b168 <stderr@@GLIBC_2.2.5+0xb18>
  404e34:	00 00 00 
  404e37:	c7 05 23 63 21 00 00 	movl   $0x0,0x216323(%rip)        # 61b164 <stderr@@GLIBC_2.2.5+0xb14>
  404e3e:	00 00 00 
  404e41:	c7 05 15 63 21 00 00 	movl   $0x0,0x216315(%rip)        # 61b160 <stderr@@GLIBC_2.2.5+0xb10>
  404e48:	00 00 00 
  404e4b:	c7 05 17 63 21 00 00 	movl   $0x0,0x216317(%rip)        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  404e52:	00 00 00 
  404e55:	c7 05 fd 62 21 00 00 	movl   $0x0,0x2162fd(%rip)        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  404e5c:	00 00 00 
  404e5f:	c7 05 ef 62 21 00 00 	movl   $0x0,0x2162ef(%rip)        # 61b158 <stderr@@GLIBC_2.2.5+0xb08>
  404e66:	00 00 00 
  404e69:	c7 05 e1 62 21 00 00 	movl   $0x0,0x2162e1(%rip)        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  404e70:	00 00 00 
  404e73:	5b                   	pop    %rbx
  404e74:	c3                   	retq   
  404e75:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  404e7c:	00 00 00 00 
  404e80:	55                   	push   %rbp
  404e81:	53                   	push   %rbx
  404e82:	48 83 ec 08          	sub    $0x8,%rsp
  404e86:	48 8b 1d 23 63 21 00 	mov    0x216323(%rip),%rbx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404e8d:	48 89 d8             	mov    %rbx,%rax
  404e90:	48 89 dd             	mov    %rbx,%rbp
  404e93:	48 d1 e8             	shr    %rax
  404e96:	48 01 d8             	add    %rbx,%rax
  404e99:	48 3b 05 00 63 21 00 	cmp    0x216300(%rip),%rax        # 61b1a0 <stderr@@GLIBC_2.2.5+0xb50>
  404ea0:	0f 87 f2 00 00 00    	ja     404f98 <__sprintf_chk@plt+0x2708>
  404ea6:	48 85 ed             	test   %rbp,%rbp
  404ea9:	74 28                	je     404ed3 <__sprintf_chk@plt+0x2643>
  404eab:	48 8b 05 f6 62 21 00 	mov    0x2162f6(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404eb2:	48 8b 15 07 63 21 00 	mov    0x216307(%rip),%rdx        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  404eb9:	48 8d 0c e8          	lea    (%rax,%rbp,8),%rcx
  404ebd:	0f 1f 00             	nopl   (%rax)
  404ec0:	48 89 10             	mov    %rdx,(%rax)
  404ec3:	48 83 c0 08          	add    $0x8,%rax
  404ec7:	48 81 c2 c0 00 00 00 	add    $0xc0,%rdx
  404ece:	48 39 c8             	cmp    %rcx,%rax
  404ed1:	75 ed                	jne    404ec0 <__sprintf_chk@plt+0x2630>
  404ed3:	83 3d 6e 62 21 00 ff 	cmpl   $0xffffffff,0x21626e(%rip)        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  404eda:	0f 84 ac 00 00 00    	je     404f8c <__sprintf_chk@plt+0x26fc>
  404ee0:	bf 80 a6 61 00       	mov    $0x61a680,%edi
  404ee5:	e8 26 d6 ff ff       	callq  402510 <_setjmp@plt>
  404eea:	85 c0                	test   %eax,%eax
  404eec:	74 52                	je     404f40 <__sprintf_chk@plt+0x26b0>
  404eee:	44 8b 05 53 62 21 00 	mov    0x216253(%rip),%r8d        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  404ef5:	41 83 f8 03          	cmp    $0x3,%r8d
  404ef9:	0f 84 e0 00 00 00    	je     404fdf <__sprintf_chk@plt+0x274f>
  404eff:	48 8b 35 aa 62 21 00 	mov    0x2162aa(%rip),%rsi        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404f06:	48 8b 3d 9b 62 21 00 	mov    0x21629b(%rip),%rdi        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404f0d:	48 85 f6             	test   %rsi,%rsi
  404f10:	74 21                	je     404f33 <__sprintf_chk@plt+0x26a3>
  404f12:	48 8b 15 a7 62 21 00 	mov    0x2162a7(%rip),%rdx        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  404f19:	48 8d 0c f7          	lea    (%rdi,%rsi,8),%rcx
  404f1d:	48 89 f8             	mov    %rdi,%rax
  404f20:	48 89 10             	mov    %rdx,(%rax)
  404f23:	48 83 c0 08          	add    $0x8,%rax
  404f27:	48 81 c2 c0 00 00 00 	add    $0xc0,%rdx
  404f2e:	48 39 c8             	cmp    %rcx,%rax
  404f31:	75 ed                	jne    404f20 <__sprintf_chk@plt+0x2690>
  404f33:	44 89 c1             	mov    %r8d,%ecx
  404f36:	b8 01 00 00 00       	mov    $0x1,%eax
  404f3b:	eb 17                	jmp    404f54 <__sprintf_chk@plt+0x26c4>
  404f3d:	0f 1f 00             	nopl   (%rax)
  404f40:	8b 0d 02 62 21 00    	mov    0x216202(%rip),%ecx        # 61b148 <stderr@@GLIBC_2.2.5+0xaf8>
  404f46:	48 8b 35 63 62 21 00 	mov    0x216263(%rip),%rsi        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404f4d:	48 8b 3d 54 62 21 00 	mov    0x216254(%rip),%rdi        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404f54:	31 d2                	xor    %edx,%edx
  404f56:	83 f9 04             	cmp    $0x4,%ecx
  404f59:	0f 44 15 ec 61 21 00 	cmove  0x2161ec(%rip),%edx        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  404f60:	48 98                	cltq   
  404f62:	44 0f b6 05 a2 61 21 	movzbl 0x2161a2(%rip),%r8d        # 61b10c <stderr@@GLIBC_2.2.5+0xabc>
  404f69:	00 
  404f6a:	01 ca                	add    %ecx,%edx
  404f6c:	48 8d 14 50          	lea    (%rax,%rdx,2),%rdx
  404f70:	0f b6 05 d0 61 21 00 	movzbl 0x2161d0(%rip),%eax        # 61b147 <stderr@@GLIBC_2.2.5+0xaf7>
  404f77:	48 8d 04 50          	lea    (%rax,%rdx,2),%rax
  404f7b:	49 8d 04 40          	lea    (%r8,%rax,2),%rax
  404f7f:	48 8b 14 c5 00 2d 41 	mov    0x412d00(,%rax,8),%rdx
  404f86:	00 
  404f87:	e8 04 87 00 00       	callq  40d690 <__sprintf_chk@plt+0xae00>
  404f8c:	48 83 c4 08          	add    $0x8,%rsp
  404f90:	5b                   	pop    %rbx
  404f91:	5d                   	pop    %rbp
  404f92:	c3                   	retq   
  404f93:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  404f98:	48 8b 3d 09 62 21 00 	mov    0x216209(%rip),%rdi        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404f9f:	e8 4c d2 ff ff       	callq  4021f0 <free@plt>
  404fa4:	48 b8 aa aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaa,%rax
  404fab:	aa aa 0a 
  404fae:	48 39 c3             	cmp    %rax,%rbx
  404fb1:	77 45                	ja     404ff8 <__sprintf_chk@plt+0x2768>
  404fb3:	48 8d 3c 5b          	lea    (%rbx,%rbx,2),%rdi
  404fb7:	48 c1 e7 03          	shl    $0x3,%rdi
  404fbb:	e8 80 bc 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  404fc0:	48 8b 2d e9 61 21 00 	mov    0x2161e9(%rip),%rbp        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  404fc7:	48 89 05 da 61 21 00 	mov    %rax,0x2161da(%rip)        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  404fce:	48 8d 44 6d 00       	lea    0x0(%rbp,%rbp,2),%rax
  404fd3:	48 89 05 c6 61 21 00 	mov    %rax,0x2161c6(%rip)        # 61b1a0 <stderr@@GLIBC_2.2.5+0xb50>
  404fda:	e9 c7 fe ff ff       	jmpq   404ea6 <__sprintf_chk@plt+0x2616>
  404fdf:	b9 95 2c 41 00       	mov    $0x412c95,%ecx
  404fe4:	ba db 0d 00 00       	mov    $0xddb,%edx
  404fe9:	be 36 37 41 00       	mov    $0x413736,%esi
  404fee:	bf 3f 37 41 00       	mov    $0x41373f,%edi
  404ff3:	e8 58 d4 ff ff       	callq  402450 <__assert_fail@plt>
  404ff8:	e8 53 be 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  404ffd:	0f 1f 00             	nopl   (%rax)
  405000:	48 8b 36             	mov    (%rsi),%rsi
  405003:	48 8b 3f             	mov    (%rdi),%rdi
  405006:	e9 45 d5 ff ff       	jmpq   402550 <strcmp@plt>
  40500b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405010:	48 89 f0             	mov    %rsi,%rax
  405013:	48 8b 37             	mov    (%rdi),%rsi
  405016:	48 8b 38             	mov    (%rax),%rdi
  405019:	e9 32 d5 ff ff       	jmpq   402550 <strcmp@plt>
  40501e:	66 90                	xchg   %ax,%ax
  405020:	55                   	push   %rbp
  405021:	48 89 f5             	mov    %rsi,%rbp
  405024:	53                   	push   %rbx
  405025:	48 89 fb             	mov    %rdi,%rbx
  405028:	48 83 ec 08          	sub    $0x8,%rsp
  40502c:	e8 ff d1 ff ff       	callq  402230 <__errno_location@plt>
  405031:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  405037:	48 83 c4 08          	add    $0x8,%rsp
  40503b:	48 89 df             	mov    %rbx,%rdi
  40503e:	5b                   	pop    %rbx
  40503f:	48 89 ee             	mov    %rbp,%rsi
  405042:	5d                   	pop    %rbp
  405043:	e9 48 d6 ff ff       	jmpq   402690 <strcoll@plt>
  405048:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40504f:	00 
  405050:	48 8b 36             	mov    (%rsi),%rsi
  405053:	48 8b 3f             	mov    (%rdi),%rdi
  405056:	eb c8                	jmp    405020 <__sprintf_chk@plt+0x2790>
  405058:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40505f:	00 
  405060:	48 89 f0             	mov    %rsi,%rax
  405063:	48 8b 37             	mov    (%rdi),%rsi
  405066:	48 8b 38             	mov    (%rax),%rdi
  405069:	eb b5                	jmp    405020 <__sprintf_chk@plt+0x2790>
  40506b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405070:	48 89 f0             	mov    %rsi,%rax
  405073:	48 8b 37             	mov    (%rdi),%rsi
  405076:	48 8b 38             	mov    (%rax),%rdi
  405079:	e9 52 57 00 00       	jmpq   40a7d0 <__sprintf_chk@plt+0x7f40>
  40507e:	66 90                	xchg   %ax,%ax
  405080:	48 8b 36             	mov    (%rsi),%rsi
  405083:	48 8b 3f             	mov    (%rdi),%rdi
  405086:	e9 45 57 00 00       	jmpq   40a7d0 <__sprintf_chk@plt+0x7f40>
  40508b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405090:	41 57                	push   %r15
  405092:	41 56                	push   %r14
  405094:	44 0f b6 f6          	movzbl %sil,%r14d
  405098:	41 55                	push   %r13
  40509a:	49 89 fd             	mov    %rdi,%r13
  40509d:	41 54                	push   %r12
  40509f:	55                   	push   %rbp
  4050a0:	53                   	push   %rbx
  4050a1:	48 83 ec 08          	sub    $0x8,%rsp
  4050a5:	48 85 ff             	test   %rdi,%rdi
  4050a8:	74 16                	je     4050c0 <__sprintf_chk@plt+0x2830>
  4050aa:	48 83 3d 16 61 21 00 	cmpq   $0x0,0x216116(%rip)        # 61b1c8 <stderr@@GLIBC_2.2.5+0xb78>
  4050b1:	00 
  4050b2:	74 0c                	je     4050c0 <__sprintf_chk@plt+0x2830>
  4050b4:	48 89 fe             	mov    %rdi,%rsi
  4050b7:	31 d2                	xor    %edx,%edx
  4050b9:	31 ff                	xor    %edi,%edi
  4050bb:	e8 60 fc ff ff       	callq  404d20 <__sprintf_chk@plt+0x2490>
  4050c0:	48 8b 1d e9 60 21 00 	mov    0x2160e9(%rip),%rbx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  4050c7:	4c 8d 24 dd f8 ff ff 	lea    -0x8(,%rbx,8),%r12
  4050ce:	ff 
  4050cf:	eb 37                	jmp    405108 <__sprintf_chk@plt+0x2878>
  4050d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4050d8:	41 80 3f 2f          	cmpb   $0x2f,(%r15)
  4050dc:	0f 85 7e 00 00 00    	jne    405160 <__sprintf_chk@plt+0x28d0>
  4050e2:	48 8b 75 08          	mov    0x8(%rbp),%rsi
  4050e6:	44 89 f2             	mov    %r14d,%edx
  4050e9:	4c 89 ff             	mov    %r15,%rdi
  4050ec:	e8 2f fc ff ff       	callq  404d20 <__sprintf_chk@plt+0x2490>
  4050f1:	83 bd a0 00 00 00 09 	cmpl   $0x9,0xa0(%rbp)
  4050f8:	0f 84 96 00 00 00    	je     405194 <__sprintf_chk@plt+0x2904>
  4050fe:	66 90                	xchg   %ax,%ax
  405100:	48 83 eb 01          	sub    $0x1,%rbx
  405104:	49 83 ec 08          	sub    $0x8,%r12
  405108:	48 85 db             	test   %rbx,%rbx
  40510b:	0f 84 97 00 00 00    	je     4051a8 <__sprintf_chk@plt+0x2918>
  405111:	48 8b 05 90 60 21 00 	mov    0x216090(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  405118:	4a 8b 2c 20          	mov    (%rax,%r12,1),%rbp
  40511c:	8b 85 a0 00 00 00    	mov    0xa0(%rbp),%eax
  405122:	83 f8 09             	cmp    $0x9,%eax
  405125:	74 05                	je     40512c <__sprintf_chk@plt+0x289c>
  405127:	83 f8 03             	cmp    $0x3,%eax
  40512a:	75 d4                	jne    405100 <__sprintf_chk@plt+0x2870>
  40512c:	4d 85 ed             	test   %r13,%r13
  40512f:	4c 8b 7d 00          	mov    0x0(%rbp),%r15
  405133:	74 ad                	je     4050e2 <__sprintf_chk@plt+0x2852>
  405135:	4c 89 ff             	mov    %r15,%rdi
  405138:	e8 53 52 00 00       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  40513d:	80 38 2e             	cmpb   $0x2e,(%rax)
  405140:	75 96                	jne    4050d8 <__sprintf_chk@plt+0x2848>
  405142:	31 d2                	xor    %edx,%edx
  405144:	80 78 01 2e          	cmpb   $0x2e,0x1(%rax)
  405148:	0f 94 c2             	sete   %dl
  40514b:	0f b6 44 10 01       	movzbl 0x1(%rax,%rdx,1),%eax
  405150:	3c 2f                	cmp    $0x2f,%al
  405152:	74 ac                	je     405100 <__sprintf_chk@plt+0x2870>
  405154:	84 c0                	test   %al,%al
  405156:	74 a8                	je     405100 <__sprintf_chk@plt+0x2870>
  405158:	e9 7b ff ff ff       	jmpq   4050d8 <__sprintf_chk@plt+0x2848>
  40515d:	0f 1f 00             	nopl   (%rax)
  405160:	31 d2                	xor    %edx,%edx
  405162:	4c 89 fe             	mov    %r15,%rsi
  405165:	4c 89 ef             	mov    %r13,%rdi
  405168:	e8 a3 54 00 00       	callq  40a610 <__sprintf_chk@plt+0x7d80>
  40516d:	48 8b 75 08          	mov    0x8(%rbp),%rsi
  405171:	49 89 c7             	mov    %rax,%r15
  405174:	48 89 c7             	mov    %rax,%rdi
  405177:	44 89 f2             	mov    %r14d,%edx
  40517a:	e8 a1 fb ff ff       	callq  404d20 <__sprintf_chk@plt+0x2490>
  40517f:	4c 89 ff             	mov    %r15,%rdi
  405182:	e8 69 d0 ff ff       	callq  4021f0 <free@plt>
  405187:	83 bd a0 00 00 00 09 	cmpl   $0x9,0xa0(%rbp)
  40518e:	0f 85 6c ff ff ff    	jne    405100 <__sprintf_chk@plt+0x2870>
  405194:	48 89 ef             	mov    %rbp,%rdi
  405197:	e8 f4 fb ff ff       	callq  404d90 <__sprintf_chk@plt+0x2500>
  40519c:	e9 5f ff ff ff       	jmpq   405100 <__sprintf_chk@plt+0x2870>
  4051a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4051a8:	48 8b 3d 01 60 21 00 	mov    0x216001(%rip),%rdi        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  4051af:	48 85 ff             	test   %rdi,%rdi
  4051b2:	74 43                	je     4051f7 <__sprintf_chk@plt+0x2967>
  4051b4:	48 8b 35 ed 5f 21 00 	mov    0x215fed(%rip),%rsi        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  4051bb:	31 d2                	xor    %edx,%edx
  4051bd:	31 c0                	xor    %eax,%eax
  4051bf:	90                   	nop
  4051c0:	48 8b 0c c6          	mov    (%rsi,%rax,8),%rcx
  4051c4:	83 b9 a0 00 00 00 09 	cmpl   $0x9,0xa0(%rcx)
  4051cb:	48 89 0c d6          	mov    %rcx,(%rsi,%rdx,8)
  4051cf:	0f 95 c1             	setne  %cl
  4051d2:	48 83 c0 01          	add    $0x1,%rax
  4051d6:	0f b6 c9             	movzbl %cl,%ecx
  4051d9:	48 01 ca             	add    %rcx,%rdx
  4051dc:	48 39 f8             	cmp    %rdi,%rax
  4051df:	75 df                	jne    4051c0 <__sprintf_chk@plt+0x2930>
  4051e1:	48 89 15 c8 5f 21 00 	mov    %rdx,0x215fc8(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  4051e8:	48 83 c4 08          	add    $0x8,%rsp
  4051ec:	5b                   	pop    %rbx
  4051ed:	5d                   	pop    %rbp
  4051ee:	41 5c                	pop    %r12
  4051f0:	41 5d                	pop    %r13
  4051f2:	41 5e                	pop    %r14
  4051f4:	41 5f                	pop    %r15
  4051f6:	c3                   	retq   
  4051f7:	31 d2                	xor    %edx,%edx
  4051f9:	eb e6                	jmp    4051e1 <__sprintf_chk@plt+0x2951>
  4051fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405200:	55                   	push   %rbp
  405201:	48 89 f5             	mov    %rsi,%rbp
  405204:	53                   	push   %rbx
  405205:	48 89 fb             	mov    %rdi,%rbx
  405208:	48 83 ec 08          	sub    $0x8,%rsp
  40520c:	48 39 f7             	cmp    %rsi,%rdi
  40520f:	72 5a                	jb     40526b <__sprintf_chk@plt+0x29db>
  405211:	e9 86 00 00 00       	jmpq   40529c <__sprintf_chk@plt+0x2a0c>
  405216:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40521d:	00 00 00 
  405220:	31 d2                	xor    %edx,%edx
  405222:	48 89 e8             	mov    %rbp,%rax
  405225:	48 8d 73 01          	lea    0x1(%rbx),%rsi
  405229:	48 f7 f1             	div    %rcx
  40522c:	31 d2                	xor    %edx,%edx
  40522e:	48 89 c7             	mov    %rax,%rdi
  405231:	48 89 f0             	mov    %rsi,%rax
  405234:	48 f7 f1             	div    %rcx
  405237:	48 39 c7             	cmp    %rax,%rdi
  40523a:	76 6c                	jbe    4052a8 <__sprintf_chk@plt+0x2a18>
  40523c:	48 8b 3d cd 53 21 00 	mov    0x2153cd(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  405243:	48 8b 47 28          	mov    0x28(%rdi),%rax
  405247:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  40524b:	73 6c                	jae    4052b9 <__sprintf_chk@plt+0x2a29>
  40524d:	48 8d 50 01          	lea    0x1(%rax),%rdx
  405251:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  405255:	c6 00 09             	movb   $0x9,(%rax)
  405258:	48 89 d8             	mov    %rbx,%rax
  40525b:	31 d2                	xor    %edx,%edx
  40525d:	48 01 cb             	add    %rcx,%rbx
  405260:	48 f7 f1             	div    %rcx
  405263:	48 29 d3             	sub    %rdx,%rbx
  405266:	48 39 dd             	cmp    %rbx,%rbp
  405269:	76 31                	jbe    40529c <__sprintf_chk@plt+0x2a0c>
  40526b:	48 8b 0d 66 5e 21 00 	mov    0x215e66(%rip),%rcx        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  405272:	48 85 c9             	test   %rcx,%rcx
  405275:	75 a9                	jne    405220 <__sprintf_chk@plt+0x2990>
  405277:	48 83 c3 01          	add    $0x1,%rbx
  40527b:	48 8b 3d 8e 53 21 00 	mov    0x21538e(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  405282:	48 8b 57 28          	mov    0x28(%rdi),%rdx
  405286:	48 3b 57 30          	cmp    0x30(%rdi),%rdx
  40528a:	73 21                	jae    4052ad <__sprintf_chk@plt+0x2a1d>
  40528c:	48 8d 42 01          	lea    0x1(%rdx),%rax
  405290:	48 39 dd             	cmp    %rbx,%rbp
  405293:	48 89 47 28          	mov    %rax,0x28(%rdi)
  405297:	c6 02 20             	movb   $0x20,(%rdx)
  40529a:	77 cf                	ja     40526b <__sprintf_chk@plt+0x29db>
  40529c:	48 83 c4 08          	add    $0x8,%rsp
  4052a0:	5b                   	pop    %rbx
  4052a1:	5d                   	pop    %rbp
  4052a2:	c3                   	retq   
  4052a3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4052a8:	48 89 f3             	mov    %rsi,%rbx
  4052ab:	eb ce                	jmp    40527b <__sprintf_chk@plt+0x29eb>
  4052ad:	be 20 00 00 00       	mov    $0x20,%esi
  4052b2:	e8 49 d1 ff ff       	callq  402400 <__overflow@plt>
  4052b7:	eb ad                	jmp    405266 <__sprintf_chk@plt+0x29d6>
  4052b9:	be 09 00 00 00       	mov    $0x9,%esi
  4052be:	e8 3d d1 ff ff       	callq  402400 <__overflow@plt>
  4052c3:	48 8b 0d 0e 5e 21 00 	mov    0x215e0e(%rip),%rcx        # 61b0d8 <stderr@@GLIBC_2.2.5+0xa88>
  4052ca:	eb 8c                	jmp    405258 <__sprintf_chk@plt+0x29c8>
  4052cc:	0f 1f 40 00          	nopl   0x0(%rax)
  4052d0:	55                   	push   %rbp
  4052d1:	49 89 d0             	mov    %rdx,%r8
  4052d4:	48 89 e5             	mov    %rsp,%rbp
  4052d7:	41 57                	push   %r15
  4052d9:	41 56                	push   %r14
  4052db:	49 89 f6             	mov    %rsi,%r14
  4052de:	41 55                	push   %r13
  4052e0:	49 89 d5             	mov    %rdx,%r13
  4052e3:	48 89 f2             	mov    %rsi,%rdx
  4052e6:	be 00 20 00 00       	mov    $0x2000,%esi
  4052eb:	41 54                	push   %r12
  4052ed:	53                   	push   %rbx
  4052ee:	48 81 ec 58 20 00 00 	sub    $0x2058,%rsp
  4052f5:	48 89 bd 88 df ff ff 	mov    %rdi,-0x2078(%rbp)
  4052fc:	48 8d bd c0 df ff ff 	lea    -0x2040(%rbp),%rdi
  405303:	48 89 8d 90 df ff ff 	mov    %rcx,-0x2070(%rbp)
  40530a:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  405311:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  405318:	00 00 
  40531a:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  40531e:	31 c0                	xor    %eax,%eax
  405320:	e8 cb 93 00 00       	callq  40e6f0 <__sprintf_chk@plt+0xbe60>
  405325:	48 89 c3             	mov    %rax,%rbx
  405328:	48 8d 85 c0 df ff ff 	lea    -0x2040(%rbp),%rax
  40532f:	48 81 fb ff 1f 00 00 	cmp    $0x1fff,%rbx
  405336:	48 89 85 98 df ff ff 	mov    %rax,-0x2068(%rbp)
  40533d:	0f 87 4d 02 00 00    	ja     405590 <__sprintf_chk@plt+0x2d00>
  405343:	80 3d a6 5d 21 00 00 	cmpb   $0x0,0x215da6(%rip)        # 61b0f0 <stderr@@GLIBC_2.2.5+0xaa0>
  40534a:	0f 85 c2 00 00 00    	jne    405412 <__sprintf_chk@plt+0x2b82>
  405350:	48 83 bd 90 df ff ff 	cmpq   $0x0,-0x2070(%rbp)
  405357:	00 
  405358:	74 1f                	je     405379 <__sprintf_chk@plt+0x2ae9>
  40535a:	e8 11 d0 ff ff       	callq  402370 <__ctype_get_mb_cur_max@plt>
  40535f:	48 83 f8 01          	cmp    $0x1,%rax
  405363:	76 68                	jbe    4053cd <__sprintf_chk@plt+0x2b3d>
  405365:	48 8b bd 98 df ff ff 	mov    -0x2068(%rbp),%rdi
  40536c:	31 d2                	xor    %edx,%edx
  40536e:	48 89 de             	mov    %rbx,%rsi
  405371:	e8 ca 7e 00 00       	callq  40d240 <__sprintf_chk@plt+0xa9b0>
  405376:	4c 63 e0             	movslq %eax,%r12
  405379:	48 8b 8d 88 df ff ff 	mov    -0x2078(%rbp),%rcx
  405380:	48 85 c9             	test   %rcx,%rcx
  405383:	74 14                	je     405399 <__sprintf_chk@plt+0x2b09>
  405385:	48 8b bd 98 df ff ff 	mov    -0x2068(%rbp),%rdi
  40538c:	48 89 da             	mov    %rbx,%rdx
  40538f:	be 01 00 00 00       	mov    $0x1,%esi
  405394:	e8 27 d3 ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  405399:	48 8b 85 90 df ff ff 	mov    -0x2070(%rbp),%rax
  4053a0:	48 85 c0             	test   %rax,%rax
  4053a3:	74 03                	je     4053a8 <__sprintf_chk@plt+0x2b18>
  4053a5:	4c 89 20             	mov    %r12,(%rax)
  4053a8:	48 89 d8             	mov    %rbx,%rax
  4053ab:	48 8b 5d c8          	mov    -0x38(%rbp),%rbx
  4053af:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
  4053b6:	00 00 
  4053b8:	0f 85 5e 02 00 00    	jne    40561c <__sprintf_chk@plt+0x2d8c>
  4053be:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  4053c2:	5b                   	pop    %rbx
  4053c3:	41 5c                	pop    %r12
  4053c5:	41 5d                	pop    %r13
  4053c7:	41 5e                	pop    %r14
  4053c9:	41 5f                	pop    %r15
  4053cb:	5d                   	pop    %rbp
  4053cc:	c3                   	retq   
  4053cd:	4c 8b b5 98 df ff ff 	mov    -0x2068(%rbp),%r14
  4053d4:	4d 8d 2c 1e          	lea    (%r14,%rbx,1),%r13
  4053d8:	4d 39 ee             	cmp    %r13,%r14
  4053db:	0f 83 29 02 00 00    	jae    40560a <__sprintf_chk@plt+0x2d7a>
  4053e1:	e8 9a d4 ff ff       	callq  402880 <__ctype_b_loc@plt>
  4053e6:	45 31 e4             	xor    %r12d,%r12d
  4053e9:	48 8b 08             	mov    (%rax),%rcx
  4053ec:	4c 89 f0             	mov    %r14,%rax
  4053ef:	90                   	nop
  4053f0:	0f b6 10             	movzbl (%rax),%edx
  4053f3:	0f b7 14 51          	movzwl (%rcx,%rdx,2),%edx
  4053f7:	66 81 e2 00 40       	and    $0x4000,%dx
  4053fc:	66 83 fa 01          	cmp    $0x1,%dx
  405400:	49 83 dc ff          	sbb    $0xffffffffffffffff,%r12
  405404:	48 83 c0 01          	add    $0x1,%rax
  405408:	4c 39 e8             	cmp    %r13,%rax
  40540b:	75 e3                	jne    4053f0 <__sprintf_chk@plt+0x2b60>
  40540d:	e9 67 ff ff ff       	jmpq   405379 <__sprintf_chk@plt+0x2ae9>
  405412:	e8 59 cf ff ff       	callq  402370 <__ctype_get_mb_cur_max@plt>
  405417:	48 83 f8 01          	cmp    $0x1,%rax
  40541b:	0f 86 b0 01 00 00    	jbe    4055d1 <__sprintf_chk@plt+0x2d41>
  405421:	4c 8b bd 98 df ff ff 	mov    -0x2068(%rbp),%r15
  405428:	4d 8d 2c 1f          	lea    (%r15,%rbx,1),%r13
  40542c:	4d 39 ef             	cmp    %r13,%r15
  40542f:	0f 83 dd 01 00 00    	jae    405612 <__sprintf_chk@plt+0x2d82>
  405435:	4c 89 fb             	mov    %r15,%rbx
  405438:	45 31 e4             	xor    %r12d,%r12d
  40543b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405440:	41 0f b6 07          	movzbl (%r15),%eax
  405444:	3c 3f                	cmp    $0x3f,%al
  405446:	0f 8f d4 00 00 00    	jg     405520 <__sprintf_chk@plt+0x2c90>
  40544c:	3c 25                	cmp    $0x25,%al
  40544e:	0f 8d e4 00 00 00    	jge    405538 <__sprintf_chk@plt+0x2ca8>
  405454:	8d 50 e0             	lea    -0x20(%rax),%edx
  405457:	80 fa 03             	cmp    $0x3,%dl
  40545a:	0f 86 d8 00 00 00    	jbe    405538 <__sprintf_chk@plt+0x2ca8>
  405460:	48 c7 85 b0 df ff ff 	movq   $0x0,-0x2050(%rbp)
  405467:	00 00 00 00 
  40546b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405470:	4c 89 ea             	mov    %r13,%rdx
  405473:	48 8d 8d b0 df ff ff 	lea    -0x2050(%rbp),%rcx
  40547a:	48 8d bd ac df ff ff 	lea    -0x2054(%rbp),%rdi
  405481:	4c 29 fa             	sub    %r15,%rdx
  405484:	4c 89 fe             	mov    %r15,%rsi
  405487:	e8 34 cf ff ff       	callq  4023c0 <mbrtowc@plt>
  40548c:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  405490:	49 89 c6             	mov    %rax,%r14
  405493:	0f 84 b8 00 00 00    	je     405551 <__sprintf_chk@plt+0x2cc1>
  405499:	48 83 f8 fe          	cmp    $0xfffffffffffffffe,%rax
  40549d:	0f 84 c8 00 00 00    	je     40556b <__sprintf_chk@plt+0x2cdb>
  4054a3:	8b bd ac df ff ff    	mov    -0x2054(%rbp),%edi
  4054a9:	48 85 c0             	test   %rax,%rax
  4054ac:	b8 01 00 00 00       	mov    $0x1,%eax
  4054b1:	4c 0f 44 f0          	cmove  %rax,%r14
  4054b5:	e8 76 d1 ff ff       	callq  402630 <wcwidth@plt>
  4054ba:	85 c0                	test   %eax,%eax
  4054bc:	78 52                	js     405510 <__sprintf_chk@plt+0x2c80>
  4054be:	4b 8d 34 37          	lea    (%r15,%r14,1),%rsi
  4054c2:	48 89 da             	mov    %rbx,%rdx
  4054c5:	0f 1f 00             	nopl   (%rax)
  4054c8:	49 83 c7 01          	add    $0x1,%r15
  4054cc:	41 0f b6 4f ff       	movzbl -0x1(%r15),%ecx
  4054d1:	48 83 c2 01          	add    $0x1,%rdx
  4054d5:	49 39 f7             	cmp    %rsi,%r15
  4054d8:	88 4a ff             	mov    %cl,-0x1(%rdx)
  4054db:	75 eb                	jne    4054c8 <__sprintf_chk@plt+0x2c38>
  4054dd:	48 98                	cltq   
  4054df:	4c 01 f3             	add    %r14,%rbx
  4054e2:	49 01 c4             	add    %rax,%r12
  4054e5:	48 8d bd b0 df ff ff 	lea    -0x2050(%rbp),%rdi
  4054ec:	e8 3f d3 ff ff       	callq  402830 <mbsinit@plt>
  4054f1:	85 c0                	test   %eax,%eax
  4054f3:	0f 84 77 ff ff ff    	je     405470 <__sprintf_chk@plt+0x2be0>
  4054f9:	4d 39 ef             	cmp    %r13,%r15
  4054fc:	0f 82 3e ff ff ff    	jb     405440 <__sprintf_chk@plt+0x2bb0>
  405502:	48 2b 9d 98 df ff ff 	sub    -0x2068(%rbp),%rbx
  405509:	e9 6b fe ff ff       	jmpq   405379 <__sprintf_chk@plt+0x2ae9>
  40550e:	66 90                	xchg   %ax,%ax
  405510:	c6 03 3f             	movb   $0x3f,(%rbx)
  405513:	4d 01 f7             	add    %r14,%r15
  405516:	49 83 c4 01          	add    $0x1,%r12
  40551a:	48 83 c3 01          	add    $0x1,%rbx
  40551e:	eb c5                	jmp    4054e5 <__sprintf_chk@plt+0x2c55>
  405520:	3c 41                	cmp    $0x41,%al
  405522:	0f 8c 38 ff ff ff    	jl     405460 <__sprintf_chk@plt+0x2bd0>
  405528:	3c 5f                	cmp    $0x5f,%al
  40552a:	7e 0c                	jle    405538 <__sprintf_chk@plt+0x2ca8>
  40552c:	8d 50 9f             	lea    -0x61(%rax),%edx
  40552f:	80 fa 1d             	cmp    $0x1d,%dl
  405532:	0f 87 28 ff ff ff    	ja     405460 <__sprintf_chk@plt+0x2bd0>
  405538:	49 83 c7 01          	add    $0x1,%r15
  40553c:	88 03                	mov    %al,(%rbx)
  40553e:	49 83 c4 01          	add    $0x1,%r12
  405542:	48 83 c3 01          	add    $0x1,%rbx
  405546:	4d 39 ef             	cmp    %r13,%r15
  405549:	0f 82 f1 fe ff ff    	jb     405440 <__sprintf_chk@plt+0x2bb0>
  40554f:	eb b1                	jmp    405502 <__sprintf_chk@plt+0x2c72>
  405551:	49 83 c7 01          	add    $0x1,%r15
  405555:	c6 03 3f             	movb   $0x3f,(%rbx)
  405558:	49 83 c4 01          	add    $0x1,%r12
  40555c:	48 83 c3 01          	add    $0x1,%rbx
  405560:	4d 39 ef             	cmp    %r13,%r15
  405563:	0f 82 d7 fe ff ff    	jb     405440 <__sprintf_chk@plt+0x2bb0>
  405569:	eb 97                	jmp    405502 <__sprintf_chk@plt+0x2c72>
  40556b:	4d 89 ef             	mov    %r13,%r15
  40556e:	c6 03 3f             	movb   $0x3f,(%rbx)
  405571:	49 83 c4 01          	add    $0x1,%r12
  405575:	48 83 c3 01          	add    $0x1,%rbx
  405579:	4d 39 ef             	cmp    %r13,%r15
  40557c:	0f 82 be fe ff ff    	jb     405440 <__sprintf_chk@plt+0x2bb0>
  405582:	e9 7b ff ff ff       	jmpq   405502 <__sprintf_chk@plt+0x2c72>
  405587:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40558e:	00 00 
  405590:	48 8d 43 1f          	lea    0x1f(%rbx),%rax
  405594:	48 8d 73 01          	lea    0x1(%rbx),%rsi
  405598:	4d 89 e8             	mov    %r13,%r8
  40559b:	48 c7 c1 ff ff ff ff 	mov    $0xffffffffffffffff,%rcx
  4055a2:	4c 89 f2             	mov    %r14,%rdx
  4055a5:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
  4055a9:	48 29 c4             	sub    %rax,%rsp
  4055ac:	48 8d 44 24 0f       	lea    0xf(%rsp),%rax
  4055b1:	48 89 85 98 df ff ff 	mov    %rax,-0x2068(%rbp)
  4055b8:	48 83 a5 98 df ff ff 	andq   $0xfffffffffffffff0,-0x2068(%rbp)
  4055bf:	f0 
  4055c0:	48 8b bd 98 df ff ff 	mov    -0x2068(%rbp),%rdi
  4055c7:	e8 24 91 00 00       	callq  40e6f0 <__sprintf_chk@plt+0xbe60>
  4055cc:	e9 72 fd ff ff       	jmpq   405343 <__sprintf_chk@plt+0x2ab3>
  4055d1:	4c 8b b5 98 df ff ff 	mov    -0x2068(%rbp),%r14
  4055d8:	4d 8d 24 1e          	lea    (%r14,%rbx,1),%r12
  4055dc:	4d 39 e6             	cmp    %r12,%r14
  4055df:	73 21                	jae    405602 <__sprintf_chk@plt+0x2d72>
  4055e1:	e8 9a d2 ff ff       	callq  402880 <__ctype_b_loc@plt>
  4055e6:	4c 89 f2             	mov    %r14,%rdx
  4055e9:	0f b6 32             	movzbl (%rdx),%esi
  4055ec:	48 8b 08             	mov    (%rax),%rcx
  4055ef:	f6 44 71 01 40       	testb  $0x40,0x1(%rcx,%rsi,2)
  4055f4:	75 03                	jne    4055f9 <__sprintf_chk@plt+0x2d69>
  4055f6:	c6 02 3f             	movb   $0x3f,(%rdx)
  4055f9:	48 83 c2 01          	add    $0x1,%rdx
  4055fd:	4c 39 e2             	cmp    %r12,%rdx
  405600:	75 e7                	jne    4055e9 <__sprintf_chk@plt+0x2d59>
  405602:	49 89 dc             	mov    %rbx,%r12
  405605:	e9 6f fd ff ff       	jmpq   405379 <__sprintf_chk@plt+0x2ae9>
  40560a:	45 31 e4             	xor    %r12d,%r12d
  40560d:	e9 67 fd ff ff       	jmpq   405379 <__sprintf_chk@plt+0x2ae9>
  405612:	31 db                	xor    %ebx,%ebx
  405614:	45 31 e4             	xor    %r12d,%r12d
  405617:	e9 5d fd ff ff       	jmpq   405379 <__sprintf_chk@plt+0x2ae9>
  40561c:	e8 7f cd ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  405621:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  405628:	0f 1f 84 00 00 00 00 
  40562f:	00 
  405630:	41 54                	push   %r12
  405632:	55                   	push   %rbp
  405633:	53                   	push   %rbx
  405634:	48 8b 46 18          	mov    0x18(%rsi),%rax
  405638:	48 8b 6e 10          	mov    0x10(%rsi),%rbp
  40563c:	49 89 c4             	mov    %rax,%r12
  40563f:	49 29 ec             	sub    %rbp,%r12
  405642:	41 c1 ec 03          	shr    $0x3,%r12d
  405646:	4d 85 e4             	test   %r12,%r12
  405649:	0f 84 8c 00 00 00    	je     4056db <__sprintf_chk@plt+0x2e4b>
  40564f:	48 39 e8             	cmp    %rbp,%rax
  405652:	0f 84 8e 00 00 00    	je     4056e6 <__sprintf_chk@plt+0x2e56>
  405658:	48 63 4e 30          	movslq 0x30(%rsi),%rcx
  40565c:	89 ca                	mov    %ecx,%edx
  40565e:	48 01 c8             	add    %rcx,%rax
  405661:	48 8b 4e 20          	mov    0x20(%rsi),%rcx
  405665:	f7 d2                	not    %edx
  405667:	48 63 d2             	movslq %edx,%rdx
  40566a:	48 21 c2             	and    %rax,%rdx
  40566d:	48 8b 46 08          	mov    0x8(%rsi),%rax
  405671:	48 89 cb             	mov    %rcx,%rbx
  405674:	48 89 56 18          	mov    %rdx,0x18(%rsi)
  405678:	48 29 c3             	sub    %rax,%rbx
  40567b:	48 29 c2             	sub    %rax,%rdx
  40567e:	48 39 da             	cmp    %rbx,%rdx
  405681:	7f 5d                	jg     4056e0 <__sprintf_chk@plt+0x2e50>
  405683:	48 8b 46 18          	mov    0x18(%rsi),%rax
  405687:	31 db                	xor    %ebx,%ebx
  405689:	48 89 46 10          	mov    %rax,0x10(%rsi)
  40568d:	48 8b 35 7c 4f 21 00 	mov    0x214f7c(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  405694:	e8 87 ce ff ff       	callq  402520 <fputs_unlocked@plt>
  405699:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4056a0:	48 8b 54 dd 00       	mov    0x0(%rbp,%rbx,8),%rdx
  4056a5:	31 c0                	xor    %eax,%eax
  4056a7:	be 59 37 41 00       	mov    $0x413759,%esi
  4056ac:	bf 01 00 00 00       	mov    $0x1,%edi
  4056b1:	48 83 c3 01          	add    $0x1,%rbx
  4056b5:	e8 76 d0 ff ff       	callq  402730 <__printf_chk@plt>
  4056ba:	49 39 dc             	cmp    %rbx,%r12
  4056bd:	77 e1                	ja     4056a0 <__sprintf_chk@plt+0x2e10>
  4056bf:	48 8b 3d 4a 4f 21 00 	mov    0x214f4a(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  4056c6:	48 8b 47 28          	mov    0x28(%rdi),%rax
  4056ca:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  4056ce:	73 1f                	jae    4056ef <__sprintf_chk@plt+0x2e5f>
  4056d0:	48 8d 50 01          	lea    0x1(%rax),%rdx
  4056d4:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  4056d8:	c6 00 0a             	movb   $0xa,(%rax)
  4056db:	5b                   	pop    %rbx
  4056dc:	5d                   	pop    %rbp
  4056dd:	41 5c                	pop    %r12
  4056df:	c3                   	retq   
  4056e0:	48 89 4e 18          	mov    %rcx,0x18(%rsi)
  4056e4:	eb 9d                	jmp    405683 <__sprintf_chk@plt+0x2df3>
  4056e6:	80 4e 50 02          	orb    $0x2,0x50(%rsi)
  4056ea:	e9 69 ff ff ff       	jmpq   405658 <__sprintf_chk@plt+0x2dc8>
  4056ef:	5b                   	pop    %rbx
  4056f0:	5d                   	pop    %rbp
  4056f1:	41 5c                	pop    %r12
  4056f3:	be 0a 00 00 00       	mov    $0xa,%esi
  4056f8:	e9 03 cd ff ff       	jmpq   402400 <__overflow@plt>
  4056fd:	0f 1f 00             	nopl   (%rax)
  405700:	55                   	push   %rbp
  405701:	48 89 fd             	mov    %rdi,%rbp
  405704:	53                   	push   %rbx
  405705:	89 d3                	mov    %edx,%ebx
  405707:	48 83 ec 08          	sub    $0x8,%rsp
  40570b:	48 85 ff             	test   %rdi,%rdi
  40570e:	74 70                	je     405780 <__sprintf_chk@plt+0x2ef0>
  405710:	31 f6                	xor    %esi,%esi
  405712:	e8 09 7d 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  405717:	48 8b 35 f2 4e 21 00 	mov    0x214ef2(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  40571e:	29 c3                	sub    %eax,%ebx
  405720:	b8 00 00 00 00       	mov    $0x0,%eax
  405725:	0f 48 d8             	cmovs  %eax,%ebx
  405728:	48 89 ef             	mov    %rbp,%rdi
  40572b:	e8 f0 cd ff ff       	callq  402520 <fputs_unlocked@plt>
  405730:	48 89 ef             	mov    %rbp,%rdi
  405733:	48 63 eb             	movslq %ebx,%rbp
  405736:	e8 45 cc ff ff       	callq  402380 <strlen@plt>
  40573b:	48 01 c5             	add    %rax,%rbp
  40573e:	66 90                	xchg   %ax,%ax
  405740:	48 8b 3d c9 4e 21 00 	mov    0x214ec9(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  405747:	48 8b 4f 28          	mov    0x28(%rdi),%rcx
  40574b:	48 3b 4f 30          	cmp    0x30(%rdi),%rcx
  40574f:	73 48                	jae    405799 <__sprintf_chk@plt+0x2f09>
  405751:	48 8d 51 01          	lea    0x1(%rcx),%rdx
  405755:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  405759:	c6 01 20             	movb   $0x20,(%rcx)
  40575c:	83 eb 01             	sub    $0x1,%ebx
  40575f:	83 fb ff             	cmp    $0xffffffff,%ebx
  405762:	75 dc                	jne    405740 <__sprintf_chk@plt+0x2eb0>
  405764:	48 8b 05 ad 58 21 00 	mov    0x2158ad(%rip),%rax        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  40576b:	48 8d 44 05 01       	lea    0x1(%rbp,%rax,1),%rax
  405770:	48 89 05 a1 58 21 00 	mov    %rax,0x2158a1(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  405777:	48 83 c4 08          	add    $0x8,%rsp
  40577b:	5b                   	pop    %rbx
  40577c:	5d                   	pop    %rbp
  40577d:	c3                   	retq   
  40577e:	66 90                	xchg   %ax,%ax
  405780:	48 89 f1             	mov    %rsi,%rcx
  405783:	bf 01 00 00 00       	mov    $0x1,%edi
  405788:	be 5e 37 41 00       	mov    $0x41375e,%esi
  40578d:	31 c0                	xor    %eax,%eax
  40578f:	48 63 eb             	movslq %ebx,%rbp
  405792:	e8 99 cf ff ff       	callq  402730 <__printf_chk@plt>
  405797:	eb cb                	jmp    405764 <__sprintf_chk@plt+0x2ed4>
  405799:	be 20 00 00 00       	mov    $0x20,%esi
  40579e:	e8 5d cc ff ff       	callq  402400 <__overflow@plt>
  4057a3:	eb b7                	jmp    40575c <__sprintf_chk@plt+0x2ecc>
  4057a5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  4057ac:	00 00 00 00 
  4057b0:	53                   	push   %rbx
  4057b1:	b8 64 37 41 00       	mov    $0x413764,%eax
  4057b6:	89 fb                	mov    %edi,%ebx
  4057b8:	48 83 ec 10          	sub    $0x10,%rsp
  4057bc:	84 d2                	test   %dl,%dl
  4057be:	74 0b                	je     4057cb <__sprintf_chk@plt+0x2f3b>
  4057c0:	31 c0                	xor    %eax,%eax
  4057c2:	80 3d 7c 59 21 00 00 	cmpb   $0x0,0x21597c(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  4057c9:	74 15                	je     4057e0 <__sprintf_chk@plt+0x2f50>
  4057cb:	48 83 c4 10          	add    $0x10,%rsp
  4057cf:	89 f2                	mov    %esi,%edx
  4057d1:	48 89 de             	mov    %rbx,%rsi
  4057d4:	5b                   	pop    %rbx
  4057d5:	48 89 c7             	mov    %rax,%rdi
  4057d8:	e9 23 ff ff ff       	jmpq   405700 <__sprintf_chk@plt+0x2e70>
  4057dd:	0f 1f 00             	nopl   (%rax)
  4057e0:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  4057e4:	e8 c7 71 00 00       	callq  40c9b0 <__sprintf_chk@plt+0xa120>
  4057e9:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  4057ed:	eb dc                	jmp    4057cb <__sprintf_chk@plt+0x2f3b>
  4057ef:	90                   	nop
  4057f0:	8b 05 42 58 21 00    	mov    0x215842(%rip),%eax        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  4057f6:	85 c0                	test   %eax,%eax
  4057f8:	75 0f                	jne    405809 <__sprintf_chk@plt+0x2f79>
  4057fa:	8b 05 34 58 21 00    	mov    0x215834(%rip),%eax        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  405800:	83 c0 01             	add    $0x1,%eax
  405803:	89 05 2b 58 21 00    	mov    %eax,0x21582b(%rip)        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  405809:	f3 c3                	repz retq 
  40580b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405810:	41 54                	push   %r12
  405812:	41 89 fc             	mov    %edi,%r12d
  405815:	48 89 d7             	mov    %rdx,%rdi
  405818:	55                   	push   %rbp
  405819:	53                   	push   %rbx
  40581a:	48 89 f3             	mov    %rsi,%rbx
  40581d:	e8 8e 92 00 00       	callq  40eab0 <__sprintf_chk@plt+0xc220>
  405822:	48 89 c5             	mov    %rax,%rbp
  405825:	e8 06 ca ff ff       	callq  402230 <__errno_location@plt>
  40582a:	8b 30                	mov    (%rax),%esi
  40582c:	31 ff                	xor    %edi,%edi
  40582e:	31 c0                	xor    %eax,%eax
  405830:	48 89 e9             	mov    %rbp,%rcx
  405833:	48 89 da             	mov    %rbx,%rdx
  405836:	e8 35 cf ff ff       	callq  402770 <error@plt>
  40583b:	45 84 e4             	test   %r12b,%r12b
  40583e:	74 10                	je     405850 <__sprintf_chk@plt+0x2fc0>
  405840:	c7 05 e6 57 21 00 02 	movl   $0x2,0x2157e6(%rip)        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  405847:	00 00 00 
  40584a:	5b                   	pop    %rbx
  40584b:	5d                   	pop    %rbp
  40584c:	41 5c                	pop    %r12
  40584e:	c3                   	retq   
  40584f:	90                   	nop
  405850:	8b 05 da 57 21 00    	mov    0x2157da(%rip),%eax        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  405856:	85 c0                	test   %eax,%eax
  405858:	75 f0                	jne    40584a <__sprintf_chk@plt+0x2fba>
  40585a:	5b                   	pop    %rbx
  40585b:	5d                   	pop    %rbp
  40585c:	c7 05 ca 57 21 00 01 	movl   $0x1,0x2157ca(%rip)        # 61b030 <stderr@@GLIBC_2.2.5+0x9e0>
  405863:	00 00 00 
  405866:	41 5c                	pop    %r12
  405868:	c3                   	retq   
  405869:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  405870:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405876:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40587d:	83 f8 09             	cmp    $0x9,%eax
  405880:	0f 94 c1             	sete   %cl
  405883:	83 f8 03             	cmp    $0x3,%eax
  405886:	0f 94 c0             	sete   %al
  405889:	41 83 f8 09          	cmp    $0x9,%r8d
  40588d:	0f 94 c2             	sete   %dl
  405890:	41 83 f8 03          	cmp    $0x3,%r8d
  405894:	41 0f 94 c0          	sete   %r8b
  405898:	44 09 c2             	or     %r8d,%edx
  40589b:	08 c8                	or     %cl,%al
  40589d:	75 41                	jne    4058e0 <__sprintf_chk@plt+0x3050>
  40589f:	84 c0                	test   %al,%al
  4058a1:	75 0d                	jne    4058b0 <__sprintf_chk@plt+0x3020>
  4058a3:	84 d2                	test   %dl,%dl
  4058a5:	b8 01 00 00 00       	mov    $0x1,%eax
  4058aa:	74 04                	je     4058b0 <__sprintf_chk@plt+0x3020>
  4058ac:	f3 c3                	repz retq 
  4058ae:	66 90                	xchg   %ax,%ax
  4058b0:	48 8b 4e 78          	mov    0x78(%rsi),%rcx
  4058b4:	48 39 4f 78          	cmp    %rcx,0x78(%rdi)
  4058b8:	48 8b 87 80 00 00 00 	mov    0x80(%rdi),%rax
  4058bf:	48 8b 96 80 00 00 00 	mov    0x80(%rsi),%rdx
  4058c6:	7f 20                	jg     4058e8 <__sprintf_chk@plt+0x3058>
  4058c8:	7c 26                	jl     4058f0 <__sprintf_chk@plt+0x3060>
  4058ca:	29 c2                	sub    %eax,%edx
  4058cc:	75 28                	jne    4058f6 <__sprintf_chk@plt+0x3066>
  4058ce:	48 8b 36             	mov    (%rsi),%rsi
  4058d1:	48 8b 3f             	mov    (%rdi),%rdi
  4058d4:	e9 77 cc ff ff       	jmpq   402550 <strcmp@plt>
  4058d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4058e0:	84 d2                	test   %dl,%dl
  4058e2:	75 bb                	jne    40589f <__sprintf_chk@plt+0x300f>
  4058e4:	0f 1f 40 00          	nopl   0x0(%rax)
  4058e8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4058ed:	c3                   	retq   
  4058ee:	66 90                	xchg   %ax,%ax
  4058f0:	b8 01 00 00 00       	mov    $0x1,%eax
  4058f5:	c3                   	retq   
  4058f6:	89 d0                	mov    %edx,%eax
  4058f8:	c3                   	retq   
  4058f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  405900:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405906:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40590d:	49 89 f1             	mov    %rsi,%r9
  405910:	83 f8 09             	cmp    $0x9,%eax
  405913:	0f 94 c1             	sete   %cl
  405916:	83 f8 03             	cmp    $0x3,%eax
  405919:	0f 94 c0             	sete   %al
  40591c:	41 83 f8 09          	cmp    $0x9,%r8d
  405920:	0f 94 c2             	sete   %dl
  405923:	41 83 f8 03          	cmp    $0x3,%r8d
  405927:	40 0f 94 c6          	sete   %sil
  40592b:	09 f2                	or     %esi,%edx
  40592d:	08 c8                	or     %cl,%al
  40592f:	75 3f                	jne    405970 <__sprintf_chk@plt+0x30e0>
  405931:	84 c0                	test   %al,%al
  405933:	75 13                	jne    405948 <__sprintf_chk@plt+0x30b8>
  405935:	84 d2                	test   %dl,%dl
  405937:	b8 01 00 00 00       	mov    $0x1,%eax
  40593c:	74 0a                	je     405948 <__sprintf_chk@plt+0x30b8>
  40593e:	66 90                	xchg   %ax,%ax
  405940:	f3 c3                	repz retq 
  405942:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405948:	48 8b 4f 68          	mov    0x68(%rdi),%rcx
  40594c:	49 39 49 68          	cmp    %rcx,0x68(%r9)
  405950:	49 8b 41 70          	mov    0x70(%r9),%rax
  405954:	48 8b 57 70          	mov    0x70(%rdi),%rdx
  405958:	7f 1e                	jg     405978 <__sprintf_chk@plt+0x30e8>
  40595a:	7c 24                	jl     405980 <__sprintf_chk@plt+0x30f0>
  40595c:	29 c2                	sub    %eax,%edx
  40595e:	75 26                	jne    405986 <__sprintf_chk@plt+0x30f6>
  405960:	48 8b 37             	mov    (%rdi),%rsi
  405963:	49 8b 39             	mov    (%r9),%rdi
  405966:	e9 e5 cb ff ff       	jmpq   402550 <strcmp@plt>
  40596b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405970:	84 d2                	test   %dl,%dl
  405972:	75 bd                	jne    405931 <__sprintf_chk@plt+0x30a1>
  405974:	0f 1f 40 00          	nopl   0x0(%rax)
  405978:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40597d:	c3                   	retq   
  40597e:	66 90                	xchg   %ax,%ax
  405980:	b8 01 00 00 00       	mov    $0x1,%eax
  405985:	c3                   	retq   
  405986:	89 d0                	mov    %edx,%eax
  405988:	c3                   	retq   
  405989:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  405990:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405996:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40599d:	49 89 f1             	mov    %rsi,%r9
  4059a0:	83 f8 09             	cmp    $0x9,%eax
  4059a3:	0f 94 c1             	sete   %cl
  4059a6:	83 f8 03             	cmp    $0x3,%eax
  4059a9:	0f 94 c0             	sete   %al
  4059ac:	41 83 f8 09          	cmp    $0x9,%r8d
  4059b0:	0f 94 c2             	sete   %dl
  4059b3:	41 83 f8 03          	cmp    $0x3,%r8d
  4059b7:	40 0f 94 c6          	sete   %sil
  4059bb:	09 f2                	or     %esi,%edx
  4059bd:	08 c8                	or     %cl,%al
  4059bf:	75 3f                	jne    405a00 <__sprintf_chk@plt+0x3170>
  4059c1:	84 c0                	test   %al,%al
  4059c3:	75 13                	jne    4059d8 <__sprintf_chk@plt+0x3148>
  4059c5:	84 d2                	test   %dl,%dl
  4059c7:	b8 01 00 00 00       	mov    $0x1,%eax
  4059cc:	74 0a                	je     4059d8 <__sprintf_chk@plt+0x3148>
  4059ce:	66 90                	xchg   %ax,%ax
  4059d0:	f3 c3                	repz retq 
  4059d2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4059d8:	48 8b 4f 58          	mov    0x58(%rdi),%rcx
  4059dc:	49 39 49 58          	cmp    %rcx,0x58(%r9)
  4059e0:	49 8b 41 60          	mov    0x60(%r9),%rax
  4059e4:	48 8b 57 60          	mov    0x60(%rdi),%rdx
  4059e8:	7f 1e                	jg     405a08 <__sprintf_chk@plt+0x3178>
  4059ea:	7c 24                	jl     405a10 <__sprintf_chk@plt+0x3180>
  4059ec:	29 c2                	sub    %eax,%edx
  4059ee:	75 26                	jne    405a16 <__sprintf_chk@plt+0x3186>
  4059f0:	48 8b 37             	mov    (%rdi),%rsi
  4059f3:	49 8b 39             	mov    (%r9),%rdi
  4059f6:	e9 55 cb ff ff       	jmpq   402550 <strcmp@plt>
  4059fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405a00:	84 d2                	test   %dl,%dl
  405a02:	75 bd                	jne    4059c1 <__sprintf_chk@plt+0x3131>
  405a04:	0f 1f 40 00          	nopl   0x0(%rax)
  405a08:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405a0d:	c3                   	retq   
  405a0e:	66 90                	xchg   %ax,%ax
  405a10:	b8 01 00 00 00       	mov    $0x1,%eax
  405a15:	c3                   	retq   
  405a16:	89 d0                	mov    %edx,%eax
  405a18:	c3                   	retq   
  405a19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  405a20:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405a26:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  405a2d:	83 f8 09             	cmp    $0x9,%eax
  405a30:	0f 94 c1             	sete   %cl
  405a33:	83 f8 03             	cmp    $0x3,%eax
  405a36:	0f 94 c2             	sete   %dl
  405a39:	41 83 f8 09          	cmp    $0x9,%r8d
  405a3d:	0f 94 c0             	sete   %al
  405a40:	41 83 f8 03          	cmp    $0x3,%r8d
  405a44:	41 0f 94 c0          	sete   %r8b
  405a48:	44 09 c0             	or     %r8d,%eax
  405a4b:	08 ca                	or     %cl,%dl
  405a4d:	75 21                	jne    405a70 <__sprintf_chk@plt+0x31e0>
  405a4f:	84 d2                	test   %dl,%dl
  405a51:	74 2d                	je     405a80 <__sprintf_chk@plt+0x31f0>
  405a53:	48 8b 4f 40          	mov    0x40(%rdi),%rcx
  405a57:	48 39 4e 40          	cmp    %rcx,0x40(%rsi)
  405a5b:	48 8b 06             	mov    (%rsi),%rax
  405a5e:	48 8b 17             	mov    (%rdi),%rdx
  405a61:	7f 15                	jg     405a78 <__sprintf_chk@plt+0x31e8>
  405a63:	7c 1f                	jl     405a84 <__sprintf_chk@plt+0x31f4>
  405a65:	48 89 d6             	mov    %rdx,%rsi
  405a68:	48 89 c7             	mov    %rax,%rdi
  405a6b:	e9 b0 f5 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  405a70:	84 c0                	test   %al,%al
  405a72:	75 db                	jne    405a4f <__sprintf_chk@plt+0x31bf>
  405a74:	0f 1f 40 00          	nopl   0x0(%rax)
  405a78:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405a7d:	c3                   	retq   
  405a7e:	66 90                	xchg   %ax,%ax
  405a80:	84 c0                	test   %al,%al
  405a82:	74 cf                	je     405a53 <__sprintf_chk@plt+0x31c3>
  405a84:	b8 01 00 00 00       	mov    $0x1,%eax
  405a89:	c3                   	retq   
  405a8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405a90:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405a96:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  405a9d:	83 f8 09             	cmp    $0x9,%eax
  405aa0:	0f 94 c1             	sete   %cl
  405aa3:	83 f8 03             	cmp    $0x3,%eax
  405aa6:	0f 94 c0             	sete   %al
  405aa9:	41 83 f8 09          	cmp    $0x9,%r8d
  405aad:	0f 94 c2             	sete   %dl
  405ab0:	41 83 f8 03          	cmp    $0x3,%r8d
  405ab4:	41 0f 94 c0          	sete   %r8b
  405ab8:	44 09 c2             	or     %r8d,%edx
  405abb:	08 c8                	or     %cl,%al
  405abd:	75 21                	jne    405ae0 <__sprintf_chk@plt+0x3250>
  405abf:	84 c0                	test   %al,%al
  405ac1:	75 0d                	jne    405ad0 <__sprintf_chk@plt+0x3240>
  405ac3:	84 d2                	test   %dl,%dl
  405ac5:	b8 01 00 00 00       	mov    $0x1,%eax
  405aca:	74 04                	je     405ad0 <__sprintf_chk@plt+0x3240>
  405acc:	f3 c3                	repz retq 
  405ace:	66 90                	xchg   %ax,%ax
  405ad0:	48 8b 36             	mov    (%rsi),%rsi
  405ad3:	48 8b 3f             	mov    (%rdi),%rdi
  405ad6:	e9 45 f5 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  405adb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405ae0:	84 d2                	test   %dl,%dl
  405ae2:	75 db                	jne    405abf <__sprintf_chk@plt+0x322f>
  405ae4:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405ae9:	c3                   	retq   
  405aea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405af0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405af6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  405afd:	83 f8 09             	cmp    $0x9,%eax
  405b00:	0f 94 c1             	sete   %cl
  405b03:	83 f8 03             	cmp    $0x3,%eax
  405b06:	0f 94 c2             	sete   %dl
  405b09:	41 83 f8 09          	cmp    $0x9,%r8d
  405b0d:	0f 94 c0             	sete   %al
  405b10:	41 83 f8 03          	cmp    $0x3,%r8d
  405b14:	41 0f 94 c0          	sete   %r8b
  405b18:	44 09 c0             	or     %r8d,%eax
  405b1b:	08 ca                	or     %cl,%dl
  405b1d:	75 21                	jne    405b40 <__sprintf_chk@plt+0x32b0>
  405b1f:	84 d2                	test   %dl,%dl
  405b21:	74 0d                	je     405b30 <__sprintf_chk@plt+0x32a0>
  405b23:	48 8b 36             	mov    (%rsi),%rsi
  405b26:	48 8b 3f             	mov    (%rdi),%rdi
  405b29:	e9 22 ca ff ff       	jmpq   402550 <strcmp@plt>
  405b2e:	66 90                	xchg   %ax,%ax
  405b30:	84 c0                	test   %al,%al
  405b32:	74 ef                	je     405b23 <__sprintf_chk@plt+0x3293>
  405b34:	b8 01 00 00 00       	mov    $0x1,%eax
  405b39:	c3                   	retq   
  405b3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405b40:	84 c0                	test   %al,%al
  405b42:	75 db                	jne    405b1f <__sprintf_chk@plt+0x328f>
  405b44:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405b49:	c3                   	retq   
  405b4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405b50:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405b56:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  405b5d:	49 89 f1             	mov    %rsi,%r9
  405b60:	83 f8 09             	cmp    $0x9,%eax
  405b63:	0f 94 c1             	sete   %cl
  405b66:	83 f8 03             	cmp    $0x3,%eax
  405b69:	0f 94 c0             	sete   %al
  405b6c:	41 83 f8 09          	cmp    $0x9,%r8d
  405b70:	0f 94 c2             	sete   %dl
  405b73:	41 83 f8 03          	cmp    $0x3,%r8d
  405b77:	40 0f 94 c6          	sete   %sil
  405b7b:	09 f2                	or     %esi,%edx
  405b7d:	08 c8                	or     %cl,%al
  405b7f:	75 27                	jne    405ba8 <__sprintf_chk@plt+0x3318>
  405b81:	84 c0                	test   %al,%al
  405b83:	75 13                	jne    405b98 <__sprintf_chk@plt+0x3308>
  405b85:	84 d2                	test   %dl,%dl
  405b87:	b8 01 00 00 00       	mov    $0x1,%eax
  405b8c:	74 0a                	je     405b98 <__sprintf_chk@plt+0x3308>
  405b8e:	66 90                	xchg   %ax,%ax
  405b90:	f3 c3                	repz retq 
  405b92:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405b98:	48 8b 37             	mov    (%rdi),%rsi
  405b9b:	49 8b 39             	mov    (%r9),%rdi
  405b9e:	e9 7d f4 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  405ba3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405ba8:	84 d2                	test   %dl,%dl
  405baa:	75 d5                	jne    405b81 <__sprintf_chk@plt+0x32f1>
  405bac:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405bb1:	c3                   	retq   
  405bb2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  405bb9:	1f 84 00 00 00 00 00 
  405bc0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  405bc6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  405bcd:	49 89 f1             	mov    %rsi,%r9
  405bd0:	83 f8 09             	cmp    $0x9,%eax
  405bd3:	0f 94 c1             	sete   %cl
  405bd6:	83 f8 03             	cmp    $0x3,%eax
  405bd9:	0f 94 c2             	sete   %dl
  405bdc:	41 83 f8 09          	cmp    $0x9,%r8d
  405be0:	0f 94 c0             	sete   %al
  405be3:	41 83 f8 03          	cmp    $0x3,%r8d
  405be7:	40 0f 94 c6          	sete   %sil
  405beb:	09 f0                	or     %esi,%eax
  405bed:	08 ca                	or     %cl,%dl
  405bef:	75 1f                	jne    405c10 <__sprintf_chk@plt+0x3380>
  405bf1:	84 d2                	test   %dl,%dl
  405bf3:	74 0b                	je     405c00 <__sprintf_chk@plt+0x3370>
  405bf5:	48 8b 37             	mov    (%rdi),%rsi
  405bf8:	49 8b 39             	mov    (%r9),%rdi
  405bfb:	e9 50 c9 ff ff       	jmpq   402550 <strcmp@plt>
  405c00:	84 c0                	test   %al,%al
  405c02:	74 f1                	je     405bf5 <__sprintf_chk@plt+0x3365>
  405c04:	b8 01 00 00 00       	mov    $0x1,%eax
  405c09:	c3                   	retq   
  405c0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405c10:	84 c0                	test   %al,%al
  405c12:	75 dd                	jne    405bf1 <__sprintf_chk@plt+0x3361>
  405c14:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  405c19:	c3                   	retq   
  405c1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405c20:	40 84 ff             	test   %dil,%dil
  405c23:	74 1b                	je     405c40 <__sprintf_chk@plt+0x33b0>
  405c25:	89 f0                	mov    %esi,%eax
  405c27:	25 00 f0 00 00       	and    $0xf000,%eax
  405c2c:	3d 00 80 00 00       	cmp    $0x8000,%eax
  405c31:	75 6d                	jne    405ca0 <__sprintf_chk@plt+0x3410>
  405c33:	31 c0                	xor    %eax,%eax
  405c35:	83 3d f0 54 21 00 03 	cmpl   $0x3,0x2154f0(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  405c3c:	74 72                	je     405cb0 <__sprintf_chk@plt+0x3420>
  405c3e:	f3 c3                	repz retq 
  405c40:	31 c0                	xor    %eax,%eax
  405c42:	83 fa 05             	cmp    $0x5,%edx
  405c45:	74 f7                	je     405c3e <__sprintf_chk@plt+0x33ae>
  405c47:	83 fa 09             	cmp    $0x9,%edx
  405c4a:	0f 94 c1             	sete   %cl
  405c4d:	83 fa 03             	cmp    $0x3,%edx
  405c50:	0f 94 c0             	sete   %al
  405c53:	09 c1                	or     %eax,%ecx
  405c55:	84 c9                	test   %cl,%cl
  405c57:	b8 2f 00 00 00       	mov    $0x2f,%eax
  405c5c:	75 e0                	jne    405c3e <__sprintf_chk@plt+0x33ae>
  405c5e:	83 3d c7 54 21 00 01 	cmpl   $0x1,0x2154c7(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  405c65:	0f 84 85 00 00 00    	je     405cf0 <__sprintf_chk@plt+0x3460>
  405c6b:	40 84 ff             	test   %dil,%dil
  405c6e:	74 50                	je     405cc0 <__sprintf_chk@plt+0x3430>
  405c70:	81 e6 00 f0 00 00    	and    $0xf000,%esi
  405c76:	b8 40 00 00 00       	mov    $0x40,%eax
  405c7b:	81 fe 00 a0 00 00    	cmp    $0xa000,%esi
  405c81:	74 bb                	je     405c3e <__sprintf_chk@plt+0x33ae>
  405c83:	81 fe 00 10 00 00    	cmp    $0x1000,%esi
  405c89:	b8 7c 00 00 00       	mov    $0x7c,%eax
  405c8e:	74 ae                	je     405c3e <__sprintf_chk@plt+0x33ae>
  405c90:	81 fe 00 c0 00 00    	cmp    $0xc000,%esi
  405c96:	0f 94 c0             	sete   %al
  405c99:	eb 47                	jmp    405ce2 <__sprintf_chk@plt+0x3452>
  405c9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405ca0:	3d 00 40 00 00       	cmp    $0x4000,%eax
  405ca5:	0f 94 c1             	sete   %cl
  405ca8:	eb ab                	jmp    405c55 <__sprintf_chk@plt+0x33c5>
  405caa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405cb0:	83 e6 49             	and    $0x49,%esi
  405cb3:	83 fe 01             	cmp    $0x1,%esi
  405cb6:	19 c0                	sbb    %eax,%eax
  405cb8:	f7 d0                	not    %eax
  405cba:	83 e0 2a             	and    $0x2a,%eax
  405cbd:	c3                   	retq   
  405cbe:	66 90                	xchg   %ax,%ax
  405cc0:	83 fa 06             	cmp    $0x6,%edx
  405cc3:	b8 40 00 00 00       	mov    $0x40,%eax
  405cc8:	0f 84 70 ff ff ff    	je     405c3e <__sprintf_chk@plt+0x33ae>
  405cce:	83 fa 01             	cmp    $0x1,%edx
  405cd1:	b8 7c 00 00 00       	mov    $0x7c,%eax
  405cd6:	0f 84 62 ff ff ff    	je     405c3e <__sprintf_chk@plt+0x33ae>
  405cdc:	83 fa 07             	cmp    $0x7,%edx
  405cdf:	0f 94 c0             	sete   %al
  405ce2:	f7 d8                	neg    %eax
  405ce4:	83 e0 3d             	and    $0x3d,%eax
  405ce7:	c3                   	retq   
  405ce8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  405cef:	00 
  405cf0:	31 c0                	xor    %eax,%eax
  405cf2:	c3                   	retq   
  405cf3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  405cfa:	84 00 00 00 00 00 
  405d00:	53                   	push   %rbx
  405d01:	40 0f b6 ff          	movzbl %dil,%edi
  405d05:	e8 16 ff ff ff       	callq  405c20 <__sprintf_chk@plt+0x3390>
  405d0a:	84 c0                	test   %al,%al
  405d0c:	89 c3                	mov    %eax,%ebx
  405d0e:	74 23                	je     405d33 <__sprintf_chk@plt+0x34a3>
  405d10:	48 8b 3d f9 48 21 00 	mov    0x2148f9(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  405d17:	48 8b 47 28          	mov    0x28(%rdi),%rax
  405d1b:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  405d1f:	73 19                	jae    405d3a <__sprintf_chk@plt+0x34aa>
  405d21:	48 8d 50 01          	lea    0x1(%rax),%rdx
  405d25:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  405d29:	88 18                	mov    %bl,(%rax)
  405d2b:	48 83 05 e5 52 21 00 	addq   $0x1,0x2152e5(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  405d32:	01 
  405d33:	84 db                	test   %bl,%bl
  405d35:	0f 95 c0             	setne  %al
  405d38:	5b                   	pop    %rbx
  405d39:	c3                   	retq   
  405d3a:	0f b6 f3             	movzbl %bl,%esi
  405d3d:	e8 be c6 ff ff       	callq  402400 <__overflow@plt>
  405d42:	eb e7                	jmp    405d2b <__sprintf_chk@plt+0x349b>
  405d44:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  405d4b:	00 00 00 00 00 
  405d50:	55                   	push   %rbp
  405d51:	48 89 fd             	mov    %rdi,%rbp
  405d54:	53                   	push   %rbx
  405d55:	31 db                	xor    %ebx,%ebx
  405d57:	48 81 ec b8 02 00 00 	sub    $0x2b8,%rsp
  405d5e:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  405d65:	00 00 
  405d67:	48 89 84 24 a8 02 00 	mov    %rax,0x2a8(%rsp)
  405d6e:	00 
  405d6f:	31 c0                	xor    %eax,%eax
  405d71:	80 3d 9c 53 21 00 00 	cmpb   $0x0,0x21539c(%rip)        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  405d78:	74 18                	je     405d92 <__sprintf_chk@plt+0x3502>
  405d7a:	83 3d cf 53 21 00 04 	cmpl   $0x4,0x2153cf(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  405d81:	0f 84 19 01 00 00    	je     405ea0 <__sprintf_chk@plt+0x3610>
  405d87:	48 63 1d ea 53 21 00 	movslq 0x2153ea(%rip),%rbx        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  405d8e:	48 83 c3 01          	add    $0x1,%rbx
  405d92:	80 3d ab 53 21 00 00 	cmpb   $0x0,0x2153ab(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  405d99:	74 1b                	je     405db6 <__sprintf_chk@plt+0x3526>
  405d9b:	83 3d ae 53 21 00 04 	cmpl   $0x4,0x2153ae(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  405da2:	0f 84 b0 00 00 00    	je     405e58 <__sprintf_chk@plt+0x35c8>
  405da8:	48 63 05 c5 53 21 00 	movslq 0x2153c5(%rip),%rax        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  405daf:	48 83 c0 01          	add    $0x1,%rax
  405db3:	48 01 c3             	add    %rax,%rbx
  405db6:	80 3d c0 53 21 00 00 	cmpb   $0x0,0x2153c0(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  405dbd:	74 17                	je     405dd6 <__sprintf_chk@plt+0x3546>
  405dbf:	83 3d 8a 53 21 00 04 	cmpl   $0x4,0x21538a(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  405dc6:	74 78                	je     405e40 <__sprintf_chk@plt+0x35b0>
  405dc8:	48 63 05 9d 53 21 00 	movslq 0x21539d(%rip),%rax        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  405dcf:	48 83 c0 01          	add    $0x1,%rax
  405dd3:	48 01 c3             	add    %rax,%rbx
  405dd6:	48 8b 15 0b 53 21 00 	mov    0x21530b(%rip),%rdx        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  405ddd:	48 8b 75 00          	mov    0x0(%rbp),%rsi
  405de1:	48 8d 4c 24 08       	lea    0x8(%rsp),%rcx
  405de6:	31 ff                	xor    %edi,%edi
  405de8:	e8 e3 f4 ff ff       	callq  4052d0 <__sprintf_chk@plt+0x2a40>
  405ded:	8b 05 39 53 21 00    	mov    0x215339(%rip),%eax        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  405df3:	48 03 5c 24 08       	add    0x8(%rsp),%rbx
  405df8:	85 c0                	test   %eax,%eax
  405dfa:	74 20                	je     405e1c <__sprintf_chk@plt+0x358c>
  405dfc:	0f b6 bd b0 00 00 00 	movzbl 0xb0(%rbp),%edi
  405e03:	8b 95 a0 00 00 00    	mov    0xa0(%rbp),%edx
  405e09:	8b 75 28             	mov    0x28(%rbp),%esi
  405e0c:	e8 0f fe ff ff       	callq  405c20 <__sprintf_chk@plt+0x3390>
  405e11:	84 c0                	test   %al,%al
  405e13:	0f 95 c0             	setne  %al
  405e16:	0f b6 c0             	movzbl %al,%eax
  405e19:	48 01 c3             	add    %rax,%rbx
  405e1c:	48 8b 94 24 a8 02 00 	mov    0x2a8(%rsp),%rdx
  405e23:	00 
  405e24:	64 48 33 14 25 28 00 	xor    %fs:0x28,%rdx
  405e2b:	00 00 
  405e2d:	48 89 d8             	mov    %rbx,%rax
  405e30:	0f 85 89 00 00 00    	jne    405ebf <__sprintf_chk@plt+0x362f>
  405e36:	48 81 c4 b8 02 00 00 	add    $0x2b8,%rsp
  405e3d:	5b                   	pop    %rbx
  405e3e:	5d                   	pop    %rbp
  405e3f:	c3                   	retq   
  405e40:	48 8b bd a8 00 00 00 	mov    0xa8(%rbp),%rdi
  405e47:	e8 34 c5 ff ff       	callq  402380 <strlen@plt>
  405e4c:	48 83 c0 01          	add    $0x1,%rax
  405e50:	eb 81                	jmp    405dd3 <__sprintf_chk@plt+0x3543>
  405e52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405e58:	80 bd b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbp)
  405e5f:	b8 02 00 00 00       	mov    $0x2,%eax
  405e64:	0f 84 49 ff ff ff    	je     405db3 <__sprintf_chk@plt+0x3523>
  405e6a:	48 8b 7d 50          	mov    0x50(%rbp),%rdi
  405e6e:	4c 8b 05 c3 52 21 00 	mov    0x2152c3(%rip),%r8        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  405e75:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  405e7a:	8b 15 c0 52 21 00    	mov    0x2152c0(%rip),%edx        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  405e80:	b9 00 02 00 00       	mov    $0x200,%ecx
  405e85:	e8 e6 5e 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  405e8a:	48 89 c7             	mov    %rax,%rdi
  405e8d:	e8 ee c4 ff ff       	callq  402380 <strlen@plt>
  405e92:	48 83 c0 01          	add    $0x1,%rax
  405e96:	e9 18 ff ff ff       	jmpq   405db3 <__sprintf_chk@plt+0x3523>
  405e9b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405ea0:	48 8b 7f 18          	mov    0x18(%rdi),%rdi
  405ea4:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
  405ea9:	e8 c2 6e 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  405eae:	48 89 c7             	mov    %rax,%rdi
  405eb1:	e8 ca c4 ff ff       	callq  402380 <strlen@plt>
  405eb6:	48 8d 58 01          	lea    0x1(%rax),%rbx
  405eba:	e9 d3 fe ff ff       	jmpq   405d92 <__sprintf_chk@plt+0x3502>
  405ebf:	e8 dc c4 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  405ec4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  405ecb:	00 00 00 00 00 
  405ed0:	41 56                	push   %r14
  405ed2:	48 8b 15 d7 52 21 00 	mov    0x2152d7(%rip),%rdx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  405ed9:	48 8b 05 40 51 21 00 	mov    0x215140(%rip),%rax        # 61b020 <stderr@@GLIBC_2.2.5+0x9d0>
  405ee0:	41 55                	push   %r13
  405ee2:	41 54                	push   %r12
  405ee4:	48 39 d0             	cmp    %rdx,%rax
  405ee7:	41 89 fc             	mov    %edi,%r12d
  405eea:	55                   	push   %rbp
  405eeb:	53                   	push   %rbx
  405eec:	48 89 d3             	mov    %rdx,%rbx
  405eef:	48 0f 46 d8          	cmovbe %rax,%rbx
  405ef3:	48 3b 1d 66 47 21 00 	cmp    0x214766(%rip),%rbx        # 61a660 <stderr@@GLIBC_2.2.5+0x10>
  405efa:	0f 86 c8 01 00 00    	jbe    4060c8 <__sprintf_chk@plt+0x3838>
  405f00:	48 89 c2             	mov    %rax,%rdx
  405f03:	48 8b 3d 1e 51 21 00 	mov    0x21511e(%rip),%rdi        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  405f0a:	48 d1 ea             	shr    %rdx
  405f0d:	48 39 d3             	cmp    %rdx,%rbx
  405f10:	0f 82 5a 02 00 00    	jb     406170 <__sprintf_chk@plt+0x38e0>
  405f16:	48 ba aa aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaa,%rdx
  405f1d:	aa aa 0a 
  405f20:	48 39 d0             	cmp    %rdx,%rax
  405f23:	0f 87 74 02 00 00    	ja     40619d <__sprintf_chk@plt+0x390d>
  405f29:	48 8d 34 40          	lea    (%rax,%rax,2),%rsi
  405f2d:	48 c1 e6 03          	shl    $0x3,%rsi
  405f31:	e8 5a ad 00 00       	callq  410c90 <__sprintf_chk@plt+0xe400>
  405f36:	48 8b 2d e3 50 21 00 	mov    0x2150e3(%rip),%rbp        # 61b020 <stderr@@GLIBC_2.2.5+0x9d0>
  405f3d:	48 89 05 e4 50 21 00 	mov    %rax,0x2150e4(%rip)        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  405f44:	48 8b 05 15 47 21 00 	mov    0x214715(%rip),%rax        # 61a660 <stderr@@GLIBC_2.2.5+0x10>
  405f4b:	48 89 ef             	mov    %rbp,%rdi
  405f4e:	48 8d 74 05 01       	lea    0x1(%rbp,%rax,1),%rsi
  405f53:	48 29 c7             	sub    %rax,%rdi
  405f56:	48 89 f1             	mov    %rsi,%rcx
  405f59:	48 0f af cf          	imul   %rdi,%rcx
  405f5d:	48 39 f5             	cmp    %rsi,%rbp
  405f60:	0f 87 37 02 00 00    	ja     40619d <__sprintf_chk@plt+0x390d>
  405f66:	31 d2                	xor    %edx,%edx
  405f68:	48 89 c8             	mov    %rcx,%rax
  405f6b:	48 f7 f7             	div    %rdi
  405f6e:	48 39 c6             	cmp    %rax,%rsi
  405f71:	0f 85 26 02 00 00    	jne    40619d <__sprintf_chk@plt+0x390d>
  405f77:	48 d1 e9             	shr    %rcx
  405f7a:	48 b8 ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rax
  405f81:	ff ff 1f 
  405f84:	48 39 c1             	cmp    %rax,%rcx
  405f87:	0f 87 10 02 00 00    	ja     40619d <__sprintf_chk@plt+0x390d>
  405f8d:	48 8d 3c cd 00 00 00 	lea    0x0(,%rcx,8),%rdi
  405f94:	00 
  405f95:	e8 a6 ac 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  405f9a:	48 8b 0d bf 46 21 00 	mov    0x2146bf(%rip),%rcx        # 61a660 <stderr@@GLIBC_2.2.5+0x10>
  405fa1:	48 39 cd             	cmp    %rcx,%rbp
  405fa4:	76 3e                	jbe    405fe4 <__sprintf_chk@plt+0x3754>
  405fa6:	48 8b 35 7b 50 21 00 	mov    0x21507b(%rip),%rsi        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  405fad:	48 8d 14 49          	lea    (%rcx,%rcx,2),%rdx
  405fb1:	48 8d 7c 6d 00       	lea    0x0(%rbp,%rbp,2),%rdi
  405fb6:	48 8d 0c cd 08 00 00 	lea    0x8(,%rcx,8),%rcx
  405fbd:	00 
  405fbe:	48 8d 14 d6          	lea    (%rsi,%rdx,8),%rdx
  405fc2:	48 8d 34 fe          	lea    (%rsi,%rdi,8),%rsi
  405fc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  405fcd:	00 00 00 
  405fd0:	48 89 42 10          	mov    %rax,0x10(%rdx)
  405fd4:	48 83 c2 18          	add    $0x18,%rdx
  405fd8:	48 01 c8             	add    %rcx,%rax
  405fdb:	48 83 c1 08          	add    $0x8,%rcx
  405fdf:	48 39 f2             	cmp    %rsi,%rdx
  405fe2:	75 ec                	jne    405fd0 <__sprintf_chk@plt+0x3740>
  405fe4:	31 c0                	xor    %eax,%eax
  405fe6:	48 85 db             	test   %rbx,%rbx
  405fe9:	48 89 2d 70 46 21 00 	mov    %rbp,0x214670(%rip)        # 61a660 <stderr@@GLIBC_2.2.5+0x10>
  405ff0:	4c 8b 05 b9 51 21 00 	mov    0x2151b9(%rip),%r8        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  405ff7:	48 8b 35 2a 50 21 00 	mov    0x21502a(%rip),%rsi        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  405ffe:	0f 85 dc 00 00 00    	jne    4060e0 <__sprintf_chk@plt+0x3850>
  406004:	31 ed                	xor    %ebp,%ebp
  406006:	4d 85 c0             	test   %r8,%r8
  406009:	0f 84 1e 01 00 00    	je     40612d <__sprintf_chk@plt+0x389d>
  40600f:	90                   	nop
  406010:	48 8b 05 91 51 21 00 	mov    0x215191(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  406017:	48 8b 3c e8          	mov    (%rax,%rbp,8),%rdi
  40601b:	e8 30 fd ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  406020:	48 85 db             	test   %rbx,%rbx
  406023:	49 89 c3             	mov    %rax,%r11
  406026:	4c 8b 35 83 51 21 00 	mov    0x215183(%rip),%r14        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  40602d:	0f 84 ed 00 00 00    	je     406120 <__sprintf_chk@plt+0x3890>
  406033:	4c 8b 2d 8e 50 21 00 	mov    0x21508e(%rip),%r13        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  40603a:	48 8b 35 e7 4f 21 00 	mov    0x214fe7(%rip),%rsi        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  406041:	b9 01 00 00 00       	mov    $0x1,%ecx
  406046:	49 8d 7e ff          	lea    -0x1(%r14),%rdi
  40604a:	eb 5b                	jmp    4060a7 <__sprintf_chk@plt+0x3817>
  40604c:	0f 1f 40 00          	nopl   0x0(%rax)
  406050:	48 8d 04 0f          	lea    (%rdi,%rcx,1),%rax
  406054:	31 d2                	xor    %edx,%edx
  406056:	48 f7 f1             	div    %rcx
  406059:	31 d2                	xor    %edx,%edx
  40605b:	49 89 c0             	mov    %rax,%r8
  40605e:	48 89 e8             	mov    %rbp,%rax
  406061:	49 f7 f0             	div    %r8
  406064:	49 89 c8             	mov    %rcx,%r8
  406067:	49 89 c2             	mov    %rax,%r10
  40606a:	31 c0                	xor    %eax,%eax
  40606c:	4d 39 d1             	cmp    %r10,%r9
  40606f:	0f 95 c0             	setne  %al
  406072:	49 8d 14 43          	lea    (%r11,%rax,2),%rdx
  406076:	48 8b 46 10          	mov    0x10(%rsi),%rax
  40607a:	4a 8d 04 d0          	lea    (%rax,%r10,8),%rax
  40607e:	4c 8b 08             	mov    (%rax),%r9
  406081:	4c 39 ca             	cmp    %r9,%rdx
  406084:	76 14                	jbe    40609a <__sprintf_chk@plt+0x380a>
  406086:	49 89 d2             	mov    %rdx,%r10
  406089:	4d 29 ca             	sub    %r9,%r10
  40608c:	4c 01 56 08          	add    %r10,0x8(%rsi)
  406090:	48 89 10             	mov    %rdx,(%rax)
  406093:	4c 39 6e 08          	cmp    %r13,0x8(%rsi)
  406097:	0f 92 06             	setb   (%rsi)
  40609a:	48 83 c6 18          	add    $0x18,%rsi
  40609e:	48 83 c1 01          	add    $0x1,%rcx
  4060a2:	4c 39 c3             	cmp    %r8,%rbx
  4060a5:	76 79                	jbe    406120 <__sprintf_chk@plt+0x3890>
  4060a7:	80 3e 00             	cmpb   $0x0,(%rsi)
  4060aa:	4c 8d 49 ff          	lea    -0x1(%rcx),%r9
  4060ae:	49 89 c8             	mov    %rcx,%r8
  4060b1:	74 e7                	je     40609a <__sprintf_chk@plt+0x380a>
  4060b3:	45 84 e4             	test   %r12b,%r12b
  4060b6:	75 98                	jne    406050 <__sprintf_chk@plt+0x37c0>
  4060b8:	48 89 e8             	mov    %rbp,%rax
  4060bb:	31 d2                	xor    %edx,%edx
  4060bd:	49 89 c8             	mov    %rcx,%r8
  4060c0:	48 f7 f1             	div    %rcx
  4060c3:	49 89 d2             	mov    %rdx,%r10
  4060c6:	eb a2                	jmp    40606a <__sprintf_chk@plt+0x37da>
  4060c8:	31 c0                	xor    %eax,%eax
  4060ca:	48 85 db             	test   %rbx,%rbx
  4060cd:	49 89 d0             	mov    %rdx,%r8
  4060d0:	48 8b 35 51 4f 21 00 	mov    0x214f51(%rip),%rsi        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  4060d7:	0f 84 27 ff ff ff    	je     406004 <__sprintf_chk@plt+0x3774>
  4060dd:	0f 1f 00             	nopl   (%rax)
  4060e0:	48 8d 78 01          	lea    0x1(%rax),%rdi
  4060e4:	48 8b 4e 10          	mov    0x10(%rsi),%rcx
  4060e8:	c6 06 01             	movb   $0x1,(%rsi)
  4060eb:	48 8d 14 7f          	lea    (%rdi,%rdi,2),%rdx
  4060ef:	48 89 56 08          	mov    %rdx,0x8(%rsi)
  4060f3:	31 d2                	xor    %edx,%edx
  4060f5:	0f 1f 00             	nopl   (%rax)
  4060f8:	48 c7 04 d1 03 00 00 	movq   $0x3,(%rcx,%rdx,8)
  4060ff:	00 
  406100:	48 83 c2 01          	add    $0x1,%rdx
  406104:	48 39 c2             	cmp    %rax,%rdx
  406107:	76 ef                	jbe    4060f8 <__sprintf_chk@plt+0x3868>
  406109:	48 83 c6 18          	add    $0x18,%rsi
  40610d:	48 39 df             	cmp    %rbx,%rdi
  406110:	0f 84 ee fe ff ff    	je     406004 <__sprintf_chk@plt+0x3774>
  406116:	48 89 f8             	mov    %rdi,%rax
  406119:	eb c5                	jmp    4060e0 <__sprintf_chk@plt+0x3850>
  40611b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406120:	48 83 c5 01          	add    $0x1,%rbp
  406124:	4c 39 f5             	cmp    %r14,%rbp
  406127:	0f 82 e3 fe ff ff    	jb     406010 <__sprintf_chk@plt+0x3780>
  40612d:	48 83 fb 01          	cmp    $0x1,%rbx
  406131:	76 31                	jbe    406164 <__sprintf_chk@plt+0x38d4>
  406133:	48 8b 15 ee 4e 21 00 	mov    0x214eee(%rip),%rdx        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  40613a:	48 8d 04 5b          	lea    (%rbx,%rbx,2),%rax
  40613e:	48 c1 e0 03          	shl    $0x3,%rax
  406142:	80 7c 02 e8 00       	cmpb   $0x0,-0x18(%rdx,%rax,1)
  406147:	75 1b                	jne    406164 <__sprintf_chk@plt+0x38d4>
  406149:	48 01 d0             	add    %rdx,%rax
  40614c:	eb 0c                	jmp    40615a <__sprintf_chk@plt+0x38ca>
  40614e:	66 90                	xchg   %ax,%ax
  406150:	48 83 e8 18          	sub    $0x18,%rax
  406154:	80 78 e8 00          	cmpb   $0x0,-0x18(%rax)
  406158:	75 0a                	jne    406164 <__sprintf_chk@plt+0x38d4>
  40615a:	48 83 eb 01          	sub    $0x1,%rbx
  40615e:	48 83 fb 01          	cmp    $0x1,%rbx
  406162:	75 ec                	jne    406150 <__sprintf_chk@plt+0x38c0>
  406164:	48 89 d8             	mov    %rbx,%rax
  406167:	5b                   	pop    %rbx
  406168:	5d                   	pop    %rbp
  406169:	41 5c                	pop    %r12
  40616b:	41 5d                	pop    %r13
  40616d:	41 5e                	pop    %r14
  40616f:	c3                   	retq   
  406170:	48 b8 55 55 55 55 55 	movabs $0x555555555555555,%rax
  406177:	55 55 05 
  40617a:	48 39 c3             	cmp    %rax,%rbx
  40617d:	77 1e                	ja     40619d <__sprintf_chk@plt+0x390d>
  40617f:	48 8d 2c 1b          	lea    (%rbx,%rbx,1),%rbp
  406183:	48 8d 74 1d 00       	lea    0x0(%rbp,%rbx,1),%rsi
  406188:	48 c1 e6 04          	shl    $0x4,%rsi
  40618c:	e8 ff aa 00 00       	callq  410c90 <__sprintf_chk@plt+0xe400>
  406191:	48 89 05 90 4e 21 00 	mov    %rax,0x214e90(%rip)        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  406198:	e9 a7 fd ff ff       	jmpq   405f44 <__sprintf_chk@plt+0x36b4>
  40619d:	e8 ae ac 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  4061a2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4061a9:	1f 84 00 00 00 00 00 
  4061b0:	55                   	push   %rbp
  4061b1:	89 fd                	mov    %edi,%ebp
  4061b3:	53                   	push   %rbx
  4061b4:	48 83 ec 28          	sub    $0x28,%rsp
  4061b8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4061bf:	00 00 
  4061c1:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  4061c6:	31 c0                	xor    %eax,%eax
  4061c8:	80 3d 76 4f 21 00 00 	cmpb   $0x0,0x214f76(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  4061cf:	74 77                	je     406248 <__sprintf_chk@plt+0x39b8>
  4061d1:	41 89 e8             	mov    %ebp,%r8d
  4061d4:	b9 5a 37 41 00       	mov    $0x41375a,%ecx
  4061d9:	ba 15 00 00 00       	mov    $0x15,%edx
  4061de:	be 01 00 00 00       	mov    $0x1,%esi
  4061e3:	48 89 e7             	mov    %rsp,%rdi
  4061e6:	31 c0                	xor    %eax,%eax
  4061e8:	e8 a3 c6 ff ff       	callq  402890 <__sprintf_chk@plt>
  4061ed:	48 89 e3             	mov    %rsp,%rbx
  4061f0:	48 89 e0             	mov    %rsp,%rax
  4061f3:	8b 08                	mov    (%rax),%ecx
  4061f5:	48 83 c0 04          	add    $0x4,%rax
  4061f9:	8d 91 ff fe fe fe    	lea    -0x1010101(%rcx),%edx
  4061ff:	f7 d1                	not    %ecx
  406201:	21 ca                	and    %ecx,%edx
  406203:	81 e2 80 80 80 80    	and    $0x80808080,%edx
  406209:	74 e8                	je     4061f3 <__sprintf_chk@plt+0x3963>
  40620b:	89 d1                	mov    %edx,%ecx
  40620d:	c1 e9 10             	shr    $0x10,%ecx
  406210:	f7 c2 80 80 00 00    	test   $0x8080,%edx
  406216:	0f 44 d1             	cmove  %ecx,%edx
  406219:	48 8d 48 02          	lea    0x2(%rax),%rcx
  40621d:	48 0f 44 c1          	cmove  %rcx,%rax
  406221:	00 d2                	add    %dl,%dl
  406223:	48 83 d8 03          	sbb    $0x3,%rax
  406227:	89 c2                	mov    %eax,%edx
  406229:	29 da                	sub    %ebx,%edx
  40622b:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  406230:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  406237:	00 00 
  406239:	89 d0                	mov    %edx,%eax
  40623b:	75 2c                	jne    406269 <__sprintf_chk@plt+0x39d9>
  40623d:	48 83 c4 28          	add    $0x28,%rsp
  406241:	5b                   	pop    %rbx
  406242:	5d                   	pop    %rbp
  406243:	c3                   	retq   
  406244:	0f 1f 40 00          	nopl   0x0(%rax)
  406248:	e8 63 67 00 00       	callq  40c9b0 <__sprintf_chk@plt+0xa120>
  40624d:	48 85 c0             	test   %rax,%rax
  406250:	48 89 c7             	mov    %rax,%rdi
  406253:	0f 84 78 ff ff ff    	je     4061d1 <__sprintf_chk@plt+0x3941>
  406259:	31 f6                	xor    %esi,%esi
  40625b:	e8 c0 71 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  406260:	31 d2                	xor    %edx,%edx
  406262:	85 c0                	test   %eax,%eax
  406264:	0f 49 d0             	cmovns %eax,%edx
  406267:	eb c2                	jmp    40622b <__sprintf_chk@plt+0x399b>
  406269:	e8 32 c1 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  40626e:	66 90                	xchg   %ax,%ax
  406270:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  406276:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40627d:	49 89 f1             	mov    %rsi,%r9
  406280:	83 f8 09             	cmp    $0x9,%eax
  406283:	0f 94 c1             	sete   %cl
  406286:	83 f8 03             	cmp    $0x3,%eax
  406289:	0f 94 c0             	sete   %al
  40628c:	41 83 f8 09          	cmp    $0x9,%r8d
  406290:	0f 94 c2             	sete   %dl
  406293:	41 83 f8 03          	cmp    $0x3,%r8d
  406297:	40 0f 94 c6          	sete   %sil
  40629b:	09 f2                	or     %esi,%edx
  40629d:	08 c8                	or     %cl,%al
  40629f:	75 27                	jne    4062c8 <__sprintf_chk@plt+0x3a38>
  4062a1:	84 c0                	test   %al,%al
  4062a3:	75 13                	jne    4062b8 <__sprintf_chk@plt+0x3a28>
  4062a5:	84 d2                	test   %dl,%dl
  4062a7:	b8 01 00 00 00       	mov    $0x1,%eax
  4062ac:	74 0a                	je     4062b8 <__sprintf_chk@plt+0x3a28>
  4062ae:	66 90                	xchg   %ax,%ax
  4062b0:	f3 c3                	repz retq 
  4062b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4062b8:	48 8b 37             	mov    (%rdi),%rsi
  4062bb:	49 8b 39             	mov    (%r9),%rdi
  4062be:	e9 0d 45 00 00       	jmpq   40a7d0 <__sprintf_chk@plt+0x7f40>
  4062c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4062c8:	84 d2                	test   %dl,%dl
  4062ca:	75 d5                	jne    4062a1 <__sprintf_chk@plt+0x3a11>
  4062cc:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4062d1:	c3                   	retq   
  4062d2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4062d9:	1f 84 00 00 00 00 00 
  4062e0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  4062e6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  4062ed:	83 f8 09             	cmp    $0x9,%eax
  4062f0:	0f 94 c1             	sete   %cl
  4062f3:	83 f8 03             	cmp    $0x3,%eax
  4062f6:	0f 94 c0             	sete   %al
  4062f9:	41 83 f8 09          	cmp    $0x9,%r8d
  4062fd:	0f 94 c2             	sete   %dl
  406300:	41 83 f8 03          	cmp    $0x3,%r8d
  406304:	41 0f 94 c0          	sete   %r8b
  406308:	44 09 c2             	or     %r8d,%edx
  40630b:	08 c8                	or     %cl,%al
  40630d:	75 21                	jne    406330 <__sprintf_chk@plt+0x3aa0>
  40630f:	84 c0                	test   %al,%al
  406311:	75 0d                	jne    406320 <__sprintf_chk@plt+0x3a90>
  406313:	84 d2                	test   %dl,%dl
  406315:	b8 01 00 00 00       	mov    $0x1,%eax
  40631a:	74 04                	je     406320 <__sprintf_chk@plt+0x3a90>
  40631c:	f3 c3                	repz retq 
  40631e:	66 90                	xchg   %ax,%ax
  406320:	48 8b 36             	mov    (%rsi),%rsi
  406323:	48 8b 3f             	mov    (%rdi),%rdi
  406326:	e9 a5 44 00 00       	jmpq   40a7d0 <__sprintf_chk@plt+0x7f40>
  40632b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406330:	84 d2                	test   %dl,%dl
  406332:	75 db                	jne    40630f <__sprintf_chk@plt+0x3a7f>
  406334:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  406339:	c3                   	retq   
  40633a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  406340:	41 54                	push   %r12
  406342:	55                   	push   %rbp
  406343:	53                   	push   %rbx
  406344:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  40634a:	48 89 f3             	mov    %rsi,%rbx
  40634d:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  406354:	83 f8 09             	cmp    $0x9,%eax
  406357:	0f 94 c1             	sete   %cl
  40635a:	83 f8 03             	cmp    $0x3,%eax
  40635d:	0f 94 c0             	sete   %al
  406360:	41 83 f8 09          	cmp    $0x9,%r8d
  406364:	0f 94 c2             	sete   %dl
  406367:	41 83 f8 03          	cmp    $0x3,%r8d
  40636b:	40 0f 94 c6          	sete   %sil
  40636f:	09 f2                	or     %esi,%edx
  406371:	08 c8                	or     %cl,%al
  406373:	75 73                	jne    4063e8 <__sprintf_chk@plt+0x3b58>
  406375:	84 c0                	test   %al,%al
  406377:	75 17                	jne    406390 <__sprintf_chk@plt+0x3b00>
  406379:	84 d2                	test   %dl,%dl
  40637b:	b8 01 00 00 00       	mov    $0x1,%eax
  406380:	74 0e                	je     406390 <__sprintf_chk@plt+0x3b00>
  406382:	5b                   	pop    %rbx
  406383:	5d                   	pop    %rbp
  406384:	41 5c                	pop    %r12
  406386:	c3                   	retq   
  406387:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40638e:	00 00 
  406390:	4c 8b 27             	mov    (%rdi),%r12
  406393:	be 2e 00 00 00       	mov    $0x2e,%esi
  406398:	4c 89 e7             	mov    %r12,%rdi
  40639b:	e8 70 c0 ff ff       	callq  402410 <strrchr@plt>
  4063a0:	48 8b 1b             	mov    (%rbx),%rbx
  4063a3:	be 2e 00 00 00       	mov    $0x2e,%esi
  4063a8:	48 89 c5             	mov    %rax,%rbp
  4063ab:	48 89 df             	mov    %rbx,%rdi
  4063ae:	e8 5d c0 ff ff       	callq  402410 <strrchr@plt>
  4063b3:	ba 19 69 41 00       	mov    $0x416919,%edx
  4063b8:	48 85 c0             	test   %rax,%rax
  4063bb:	48 0f 44 c2          	cmove  %rdx,%rax
  4063bf:	48 85 ed             	test   %rbp,%rbp
  4063c2:	48 0f 45 d5          	cmovne %rbp,%rdx
  4063c6:	48 89 c6             	mov    %rax,%rsi
  4063c9:	48 89 d7             	mov    %rdx,%rdi
  4063cc:	e8 7f c1 ff ff       	callq  402550 <strcmp@plt>
  4063d1:	85 c0                	test   %eax,%eax
  4063d3:	75 ad                	jne    406382 <__sprintf_chk@plt+0x3af2>
  4063d5:	48 89 de             	mov    %rbx,%rsi
  4063d8:	4c 89 e7             	mov    %r12,%rdi
  4063db:	5b                   	pop    %rbx
  4063dc:	5d                   	pop    %rbp
  4063dd:	41 5c                	pop    %r12
  4063df:	e9 6c c1 ff ff       	jmpq   402550 <strcmp@plt>
  4063e4:	0f 1f 40 00          	nopl   0x0(%rax)
  4063e8:	84 d2                	test   %dl,%dl
  4063ea:	75 89                	jne    406375 <__sprintf_chk@plt+0x3ae5>
  4063ec:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4063f1:	eb 8f                	jmp    406382 <__sprintf_chk@plt+0x3af2>
  4063f3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  4063fa:	84 00 00 00 00 00 
  406400:	48 83 3d 00 40 21 00 	cmpq   $0x0,0x214000(%rip)        # 61a408 <_fini@@Base+0x20850c>
  406407:	00 
  406408:	74 0e                	je     406418 <__sprintf_chk@plt+0x3b88>
  40640a:	bf 00 a4 61 00       	mov    $0x61a400,%edi
  40640f:	eb 2f                	jmp    406440 <__sprintf_chk@plt+0x3bb0>
  406411:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406418:	48 83 ec 08          	sub    $0x8,%rsp
  40641c:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  406421:	e8 1a 00 00 00       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406426:	bf 10 a4 61 00       	mov    $0x61a410,%edi
  40642b:	e8 10 00 00 00       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406430:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  406435:	48 83 c4 08          	add    $0x8,%rsp
  406439:	eb 05                	jmp    406440 <__sprintf_chk@plt+0x3bb0>
  40643b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406440:	48 83 ec 18          	sub    $0x18,%rsp
  406444:	80 3d dd 4c 21 00 00 	cmpb   $0x0,0x214cdd(%rip)        # 61b128 <stderr@@GLIBC_2.2.5+0xad8>
  40644b:	74 23                	je     406470 <__sprintf_chk@plt+0x3be0>
  40644d:	48 8b 37             	mov    (%rdi),%rsi
  406450:	48 8b 0d b9 41 21 00 	mov    0x2141b9(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  406457:	ba 01 00 00 00       	mov    $0x1,%edx
  40645c:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  406460:	48 83 c4 18          	add    $0x18,%rsp
  406464:	e9 57 c2 ff ff       	jmpq   4026c0 <fwrite_unlocked@plt>
  406469:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406470:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  406475:	c6 05 ac 4c 21 00 01 	movb   $0x1,0x214cac(%rip)        # 61b128 <stderr@@GLIBC_2.2.5+0xad8>
  40647c:	e8 7f ff ff ff       	callq  406400 <__sprintf_chk@plt+0x3b70>
  406481:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  406486:	eb c5                	jmp    40644d <__sprintf_chk@plt+0x3bbd>
  406488:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40648f:	00 
  406490:	53                   	push   %rbx
  406491:	48 83 c4 80          	add    $0xffffffffffffff80,%rsp
  406495:	eb 58                	jmp    4064ef <__sprintf_chk@plt+0x3c5f>
  406497:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40649e:	00 00 
  4064a0:	48 8b 3d 69 41 21 00 	mov    0x214169(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  4064a7:	e8 74 c3 ff ff       	callq  402820 <fflush_unlocked@plt>
  4064ac:	31 ff                	xor    %edi,%edi
  4064ae:	48 89 e2             	mov    %rsp,%rdx
  4064b1:	be 40 b0 61 00       	mov    $0x61b040,%esi
  4064b6:	e8 15 bd ff ff       	callq  4021d0 <sigprocmask@plt>
  4064bb:	8b 1d 77 4b 21 00    	mov    0x214b77(%rip),%ebx        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  4064c1:	8b 05 6d 4b 21 00    	mov    0x214b6d(%rip),%eax        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  4064c7:	85 c0                	test   %eax,%eax
  4064c9:	74 5d                	je     406528 <__sprintf_chk@plt+0x3c98>
  4064cb:	83 e8 01             	sub    $0x1,%eax
  4064ce:	bb 13 00 00 00       	mov    $0x13,%ebx
  4064d3:	89 05 5b 4b 21 00    	mov    %eax,0x214b5b(%rip)        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  4064d9:	89 df                	mov    %ebx,%edi
  4064db:	e8 00 bd ff ff       	callq  4021e0 <raise@plt>
  4064e0:	31 d2                	xor    %edx,%edx
  4064e2:	48 89 e6             	mov    %rsp,%rsi
  4064e5:	bf 02 00 00 00       	mov    $0x2,%edi
  4064ea:	e8 e1 bc ff ff       	callq  4021d0 <sigprocmask@plt>
  4064ef:	8b 05 43 4b 21 00    	mov    0x214b43(%rip),%eax        # 61b038 <stderr@@GLIBC_2.2.5+0x9e8>
  4064f5:	85 c0                	test   %eax,%eax
  4064f7:	75 0a                	jne    406503 <__sprintf_chk@plt+0x3c73>
  4064f9:	8b 05 35 4b 21 00    	mov    0x214b35(%rip),%eax        # 61b034 <stderr@@GLIBC_2.2.5+0x9e4>
  4064ff:	85 c0                	test   %eax,%eax
  406501:	74 35                	je     406538 <__sprintf_chk@plt+0x3ca8>
  406503:	80 3d 1e 4c 21 00 00 	cmpb   $0x0,0x214c1e(%rip)        # 61b128 <stderr@@GLIBC_2.2.5+0xad8>
  40650a:	74 94                	je     4064a0 <__sprintf_chk@plt+0x3c10>
  40650c:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  406511:	e8 2a ff ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406516:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  40651b:	e8 20 ff ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406520:	e9 7b ff ff ff       	jmpq   4064a0 <__sprintf_chk@plt+0x3c10>
  406525:	0f 1f 00             	nopl   (%rax)
  406528:	31 f6                	xor    %esi,%esi
  40652a:	89 df                	mov    %ebx,%edi
  40652c:	e8 2f c0 ff ff       	callq  402560 <signal@plt>
  406531:	eb a6                	jmp    4064d9 <__sprintf_chk@plt+0x3c49>
  406533:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406538:	48 83 ec 80          	sub    $0xffffffffffffff80,%rsp
  40653c:	5b                   	pop    %rbx
  40653d:	c3                   	retq   
  40653e:	66 90                	xchg   %ax,%ax
  406540:	41 57                	push   %r15
  406542:	41 56                	push   %r14
  406544:	41 55                	push   %r13
  406546:	49 89 cd             	mov    %rcx,%r13
  406549:	41 54                	push   %r12
  40654b:	55                   	push   %rbp
  40654c:	48 89 fd             	mov    %rdi,%rbp
  40654f:	53                   	push   %rbx
  406550:	48 89 d3             	mov    %rdx,%rbx
  406553:	48 83 ec 08          	sub    $0x8,%rsp
  406557:	40 84 f6             	test   %sil,%sil
  40655a:	48 8b 17             	mov    (%rdi),%rdx
  40655d:	4c 8b 67 08          	mov    0x8(%rdi),%r12
  406561:	0f 84 56 04 00 00    	je     4069bd <__sprintf_chk@plt+0x412d>
  406567:	80 3d bb 4b 21 00 00 	cmpb   $0x0,0x214bbb(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  40656e:	75 10                	jne    406580 <__sprintf_chk@plt+0x3cf0>
  406570:	45 31 f6             	xor    %r14d,%r14d
  406573:	e9 87 01 00 00       	jmpq   4066ff <__sprintf_chk@plt+0x3e6f>
  406578:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40657f:	00 
  406580:	44 0f b6 b5 b1 00 00 	movzbl 0xb1(%rbp),%r14d
  406587:	00 
  406588:	44 8b bd a4 00 00 00 	mov    0xa4(%rbp),%r15d
  40658f:	45 84 f6             	test   %r14b,%r14b
  406592:	75 1c                	jne    4065b0 <__sprintf_chk@plt+0x3d20>
  406594:	bf 0c 00 00 00       	mov    $0xc,%edi
  406599:	e8 32 e7 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40659e:	84 c0                	test   %al,%al
  4065a0:	ba 0c 00 00 00       	mov    $0xc,%edx
  4065a5:	0f 85 05 01 00 00    	jne    4066b0 <__sprintf_chk@plt+0x3e20>
  4065ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4065b0:	80 bd b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbp)
  4065b7:	75 6f                	jne    406628 <__sprintf_chk@plt+0x3d98>
  4065b9:	8b 85 a0 00 00 00    	mov    0xa0(%rbp),%eax
  4065bf:	8b 14 85 60 2c 41 00 	mov    0x412c60(,%rax,4),%edx
  4065c6:	83 fa 05             	cmp    $0x5,%edx
  4065c9:	0f 85 b1 00 00 00    	jne    406680 <__sprintf_chk@plt+0x3df0>
  4065cf:	4c 89 e7             	mov    %r12,%rdi
  4065d2:	e8 a9 bd ff ff       	callq  402380 <strlen@plt>
  4065d7:	48 8b 2d 42 4b 21 00 	mov    0x214b42(%rip),%rbp        # 61b120 <stderr@@GLIBC_2.2.5+0xad0>
  4065de:	49 89 c6             	mov    %rax,%r14
  4065e1:	4d 8d 3c 04          	lea    (%r12,%rax,1),%r15
  4065e5:	48 85 ed             	test   %rbp,%rbp
  4065e8:	74 2f                	je     406619 <__sprintf_chk@plt+0x3d89>
  4065ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4065f0:	48 8b 55 00          	mov    0x0(%rbp),%rdx
  4065f4:	49 39 d6             	cmp    %rdx,%r14
  4065f7:	72 17                	jb     406610 <__sprintf_chk@plt+0x3d80>
  4065f9:	48 8b 75 08          	mov    0x8(%rbp),%rsi
  4065fd:	4c 89 ff             	mov    %r15,%rdi
  406600:	48 29 d7             	sub    %rdx,%rdi
  406603:	e8 38 bc ff ff       	callq  402240 <strncmp@plt>
  406608:	85 c0                	test   %eax,%eax
  40660a:	0f 84 90 02 00 00    	je     4068a0 <__sprintf_chk@plt+0x4010>
  406610:	48 8b 6d 20          	mov    0x20(%rbp),%rbp
  406614:	48 85 ed             	test   %rbp,%rbp
  406617:	75 d7                	jne    4065f0 <__sprintf_chk@plt+0x3d60>
  406619:	ba 05 00 00 00       	mov    $0x5,%edx
  40661e:	e9 8d 00 00 00       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  406623:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406628:	44 89 f8             	mov    %r15d,%eax
  40662b:	25 00 f0 00 00       	and    $0xf000,%eax
  406630:	3d 00 80 00 00       	cmp    $0x8000,%eax
  406635:	0f 84 2d 02 00 00    	je     406868 <__sprintf_chk@plt+0x3fd8>
  40663b:	3d 00 40 00 00       	cmp    $0x4000,%eax
  406640:	0f 84 9a 02 00 00    	je     4068e0 <__sprintf_chk@plt+0x4050>
  406646:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  40664b:	0f 84 08 02 00 00    	je     406859 <__sprintf_chk@plt+0x3fc9>
  406651:	3d 00 10 00 00       	cmp    $0x1000,%eax
  406656:	ba 08 00 00 00       	mov    $0x8,%edx
  40665b:	74 53                	je     4066b0 <__sprintf_chk@plt+0x3e20>
  40665d:	3d 00 c0 00 00       	cmp    $0xc000,%eax
  406662:	b2 09                	mov    $0x9,%dl
  406664:	74 4a                	je     4066b0 <__sprintf_chk@plt+0x3e20>
  406666:	3d 00 60 00 00       	cmp    $0x6000,%eax
  40666b:	b2 0a                	mov    $0xa,%dl
  40666d:	74 41                	je     4066b0 <__sprintf_chk@plt+0x3e20>
  40666f:	31 d2                	xor    %edx,%edx
  406671:	3d 00 20 00 00       	cmp    $0x2000,%eax
  406676:	0f 95 c2             	setne  %dl
  406679:	8d 54 12 0b          	lea    0xb(%rdx,%rdx,1),%edx
  40667d:	eb 31                	jmp    4066b0 <__sprintf_chk@plt+0x3e20>
  40667f:	90                   	nop
  406680:	83 fa 07             	cmp    $0x7,%edx
  406683:	0f 94 c0             	sete   %al
  406686:	41 21 c6             	and    %eax,%r14d
  406689:	45 84 f6             	test   %r14b,%r14b
  40668c:	74 22                	je     4066b0 <__sprintf_chk@plt+0x3e20>
  40668e:	80 3d 03 4b 21 00 00 	cmpb   $0x0,0x214b03(%rip)        # 61b198 <stderr@@GLIBC_2.2.5+0xb48>
  406695:	ba 0d 00 00 00       	mov    $0xd,%edx
  40669a:	75 14                	jne    4066b0 <__sprintf_chk@plt+0x3e20>
  40669c:	bf 0d 00 00 00       	mov    $0xd,%edi
  4066a1:	e8 2a e6 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4066a6:	3c 01                	cmp    $0x1,%al
  4066a8:	19 d2                	sbb    %edx,%edx
  4066aa:	83 e2 fa             	and    $0xfffffffa,%edx
  4066ad:	83 c2 0d             	add    $0xd,%edx
  4066b0:	89 d5                	mov    %edx,%ebp
  4066b2:	48 c1 e5 04          	shl    $0x4,%rbp
  4066b6:	48 81 c5 e0 a3 61 00 	add    $0x61a3e0,%rbp
  4066bd:	0f 1f 00             	nopl   (%rax)
  4066c0:	48 83 7d 08 00       	cmpq   $0x0,0x8(%rbp)
  4066c5:	bf 04 00 00 00       	mov    $0x4,%edi
  4066ca:	0f 84 c0 01 00 00    	je     406890 <__sprintf_chk@plt+0x4000>
  4066d0:	e8 fb e5 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4066d5:	84 c0                	test   %al,%al
  4066d7:	0f 85 63 01 00 00    	jne    406840 <__sprintf_chk@plt+0x3fb0>
  4066dd:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  4066e2:	41 be 01 00 00 00    	mov    $0x1,%r14d
  4066e8:	e8 53 fd ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4066ed:	48 89 ef             	mov    %rbp,%rdi
  4066f0:	e8 4b fd ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4066f5:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  4066fa:	e8 41 fd ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4066ff:	48 85 db             	test   %rbx,%rbx
  406702:	0f 84 d0 00 00 00    	je     4067d8 <__sprintf_chk@plt+0x3f48>
  406708:	80 3d 21 4a 21 00 00 	cmpb   $0x0,0x214a21(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  40670f:	74 21                	je     406732 <__sprintf_chk@plt+0x3ea2>
  406711:	48 8b 43 18          	mov    0x18(%rbx),%rax
  406715:	48 8d 50 08          	lea    0x8(%rax),%rdx
  406719:	48 39 53 20          	cmp    %rdx,0x20(%rbx)
  40671d:	0f 82 fd 00 00 00    	jb     406820 <__sprintf_chk@plt+0x3f90>
  406723:	48 8b 15 ee 48 21 00 	mov    0x2148ee(%rip),%rdx        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  40672a:	48 89 10             	mov    %rdx,(%rax)
  40672d:	48 83 43 18 08       	addq   $0x8,0x18(%rbx)
  406732:	48 8b 15 af 49 21 00 	mov    0x2149af(%rip),%rdx        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  406739:	48 8b 3d d0 3e 21 00 	mov    0x213ed0(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  406740:	31 c9                	xor    %ecx,%ecx
  406742:	4c 89 e6             	mov    %r12,%rsi
  406745:	e8 86 eb ff ff       	callq  4052d0 <__sprintf_chk@plt+0x2a40>
  40674a:	48 89 c2             	mov    %rax,%rdx
  40674d:	48 03 15 c4 48 21 00 	add    0x2148c4(%rip),%rdx        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  406754:	80 3d d5 49 21 00 00 	cmpb   $0x0,0x2149d5(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  40675b:	48 89 c5             	mov    %rax,%rbp
  40675e:	48 89 15 b3 48 21 00 	mov    %rdx,0x2148b3(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  406765:	74 1a                	je     406781 <__sprintf_chk@plt+0x3ef1>
  406767:	48 8b 43 18          	mov    0x18(%rbx),%rax
  40676b:	48 8d 48 08          	lea    0x8(%rax),%rcx
  40676f:	48 39 4b 20          	cmp    %rcx,0x20(%rbx)
  406773:	0f 82 87 00 00 00    	jb     406800 <__sprintf_chk@plt+0x3f70>
  406779:	48 89 10             	mov    %rdx,(%rax)
  40677c:	48 83 43 18 08       	addq   $0x8,0x18(%rbx)
  406781:	e8 0a fd ff ff       	callq  406490 <__sprintf_chk@plt+0x3c00>
  406786:	45 84 f6             	test   %r14b,%r14b
  406789:	75 15                	jne    4067a0 <__sprintf_chk@plt+0x3f10>
  40678b:	48 83 c4 08          	add    $0x8,%rsp
  40678f:	48 89 e8             	mov    %rbp,%rax
  406792:	5b                   	pop    %rbx
  406793:	5d                   	pop    %rbp
  406794:	41 5c                	pop    %r12
  406796:	41 5d                	pop    %r13
  406798:	41 5e                	pop    %r14
  40679a:	41 5f                	pop    %r15
  40679c:	c3                   	retq   
  40679d:	0f 1f 00             	nopl   (%rax)
  4067a0:	e8 5b fc ff ff       	callq  406400 <__sprintf_chk@plt+0x3b70>
  4067a5:	48 8b 0d 1c 49 21 00 	mov    0x21491c(%rip),%rcx        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  4067ac:	31 d2                	xor    %edx,%edx
  4067ae:	4c 89 e8             	mov    %r13,%rax
  4067b1:	48 f7 f1             	div    %rcx
  4067b4:	31 d2                	xor    %edx,%edx
  4067b6:	48 89 c6             	mov    %rax,%rsi
  4067b9:	4a 8d 44 2d ff       	lea    -0x1(%rbp,%r13,1),%rax
  4067be:	48 f7 f1             	div    %rcx
  4067c1:	48 39 c6             	cmp    %rax,%rsi
  4067c4:	74 c5                	je     40678b <__sprintf_chk@plt+0x3efb>
  4067c6:	bf 50 a5 61 00       	mov    $0x61a550,%edi
  4067cb:	e8 70 fc ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  4067d0:	eb b9                	jmp    40678b <__sprintf_chk@plt+0x3efb>
  4067d2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4067d8:	48 8b 15 09 49 21 00 	mov    0x214909(%rip),%rdx        # 61b0e8 <stderr@@GLIBC_2.2.5+0xa98>
  4067df:	48 8b 3d 2a 3e 21 00 	mov    0x213e2a(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  4067e6:	31 c9                	xor    %ecx,%ecx
  4067e8:	4c 89 e6             	mov    %r12,%rsi
  4067eb:	e8 e0 ea ff ff       	callq  4052d0 <__sprintf_chk@plt+0x2a40>
  4067f0:	48 89 c5             	mov    %rax,%rbp
  4067f3:	48 01 05 1e 48 21 00 	add    %rax,0x21481e(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  4067fa:	eb 85                	jmp    406781 <__sprintf_chk@plt+0x3ef1>
  4067fc:	0f 1f 40 00          	nopl   0x0(%rax)
  406800:	be 08 00 00 00       	mov    $0x8,%esi
  406805:	48 89 df             	mov    %rbx,%rdi
  406808:	e8 13 bf ff ff       	callq  402720 <_obstack_newchunk@plt>
  40680d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  406811:	48 8b 15 00 48 21 00 	mov    0x214800(%rip),%rdx        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  406818:	e9 5c ff ff ff       	jmpq   406779 <__sprintf_chk@plt+0x3ee9>
  40681d:	0f 1f 00             	nopl   (%rax)
  406820:	be 08 00 00 00       	mov    $0x8,%esi
  406825:	48 89 df             	mov    %rbx,%rdi
  406828:	e8 f3 be ff ff       	callq  402720 <_obstack_newchunk@plt>
  40682d:	48 8b 43 18          	mov    0x18(%rbx),%rax
  406831:	e9 ed fe ff ff       	jmpq   406723 <__sprintf_chk@plt+0x3e93>
  406836:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40683d:	00 00 00 
  406840:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  406845:	e8 f6 fb ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  40684a:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  40684f:	e8 ec fb ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406854:	e9 84 fe ff ff       	jmpq   4066dd <__sprintf_chk@plt+0x3e4d>
  406859:	ba 07 00 00 00       	mov    $0x7,%edx
  40685e:	e9 26 fe ff ff       	jmpq   406689 <__sprintf_chk@plt+0x3df9>
  406863:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406868:	41 f7 c7 00 08 00 00 	test   $0x800,%r15d
  40686f:	74 47                	je     4068b8 <__sprintf_chk@plt+0x4028>
  406871:	bf 10 00 00 00       	mov    $0x10,%edi
  406876:	e8 55 e4 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40687b:	84 c0                	test   %al,%al
  40687d:	74 39                	je     4068b8 <__sprintf_chk@plt+0x4028>
  40687f:	ba 10 00 00 00       	mov    $0x10,%edx
  406884:	e9 27 fe ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  406889:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406890:	e8 3b e4 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  406895:	41 89 c6             	mov    %eax,%r14d
  406898:	e9 62 fe ff ff       	jmpq   4066ff <__sprintf_chk@plt+0x3e6f>
  40689d:	0f 1f 00             	nopl   (%rax)
  4068a0:	48 85 ed             	test   %rbp,%rbp
  4068a3:	0f 84 70 fd ff ff    	je     406619 <__sprintf_chk@plt+0x3d89>
  4068a9:	48 83 c5 10          	add    $0x10,%rbp
  4068ad:	e9 0e fe ff ff       	jmpq   4066c0 <__sprintf_chk@plt+0x3e30>
  4068b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4068b8:	41 f7 c7 00 04 00 00 	test   $0x400,%r15d
  4068bf:	74 7a                	je     40693b <__sprintf_chk@plt+0x40ab>
  4068c1:	bf 11 00 00 00       	mov    $0x11,%edi
  4068c6:	e8 05 e4 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4068cb:	84 c0                	test   %al,%al
  4068cd:	74 6c                	je     40693b <__sprintf_chk@plt+0x40ab>
  4068cf:	ba 11 00 00 00       	mov    $0x11,%edx
  4068d4:	e9 d7 fd ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  4068d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4068e0:	44 89 f8             	mov    %r15d,%eax
  4068e3:	25 02 02 00 00       	and    $0x202,%eax
  4068e8:	3d 02 02 00 00       	cmp    $0x202,%eax
  4068ed:	0f 84 ae 00 00 00    	je     4069a1 <__sprintf_chk@plt+0x4111>
  4068f3:	41 f6 c7 02          	test   $0x2,%r15b
  4068f7:	74 17                	je     406910 <__sprintf_chk@plt+0x4080>
  4068f9:	bf 13 00 00 00       	mov    $0x13,%edi
  4068fe:	e8 cd e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  406903:	84 c0                	test   %al,%al
  406905:	ba 13 00 00 00       	mov    $0x13,%edx
  40690a:	0f 85 a0 fd ff ff    	jne    4066b0 <__sprintf_chk@plt+0x3e20>
  406910:	41 81 e7 00 02 00 00 	and    $0x200,%r15d
  406917:	ba 06 00 00 00       	mov    $0x6,%edx
  40691c:	0f 84 8e fd ff ff    	je     4066b0 <__sprintf_chk@plt+0x3e20>
  406922:	bf 12 00 00 00       	mov    $0x12,%edi
  406927:	e8 a4 e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40692c:	3c 01                	cmp    $0x1,%al
  40692e:	19 d2                	sbb    %edx,%edx
  406930:	83 e2 f4             	and    $0xfffffff4,%edx
  406933:	83 c2 12             	add    $0x12,%edx
  406936:	e9 75 fd ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  40693b:	bf 15 00 00 00       	mov    $0x15,%edi
  406940:	e8 8b e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  406945:	84 c0                	test   %al,%al
  406947:	74 13                	je     40695c <__sprintf_chk@plt+0x40cc>
  406949:	80 bd b8 00 00 00 00 	cmpb   $0x0,0xb8(%rbp)
  406950:	74 0a                	je     40695c <__sprintf_chk@plt+0x40cc>
  406952:	ba 15 00 00 00       	mov    $0x15,%edx
  406957:	e9 54 fd ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  40695c:	41 83 e7 49          	and    $0x49,%r15d
  406960:	74 18                	je     40697a <__sprintf_chk@plt+0x40ea>
  406962:	bf 0e 00 00 00       	mov    $0xe,%edi
  406967:	e8 64 e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40696c:	84 c0                	test   %al,%al
  40696e:	74 0a                	je     40697a <__sprintf_chk@plt+0x40ea>
  406970:	ba 0e 00 00 00       	mov    $0xe,%edx
  406975:	e9 36 fd ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  40697a:	48 83 7d 20 01       	cmpq   $0x1,0x20(%rbp)
  40697f:	0f 86 4a fc ff ff    	jbe    4065cf <__sprintf_chk@plt+0x3d3f>
  406985:	bf 16 00 00 00       	mov    $0x16,%edi
  40698a:	e8 41 e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40698f:	84 c0                	test   %al,%al
  406991:	0f 84 38 fc ff ff    	je     4065cf <__sprintf_chk@plt+0x3d3f>
  406997:	ba 16 00 00 00       	mov    $0x16,%edx
  40699c:	e9 0f fd ff ff       	jmpq   4066b0 <__sprintf_chk@plt+0x3e20>
  4069a1:	bf 14 00 00 00       	mov    $0x14,%edi
  4069a6:	e8 25 e3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4069ab:	84 c0                	test   %al,%al
  4069ad:	ba 14 00 00 00       	mov    $0x14,%edx
  4069b2:	0f 85 f8 fc ff ff    	jne    4066b0 <__sprintf_chk@plt+0x3e20>
  4069b8:	e9 36 ff ff ff       	jmpq   4068f3 <__sprintf_chk@plt+0x4063>
  4069bd:	80 3d 65 47 21 00 00 	cmpb   $0x0,0x214765(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  4069c4:	75 12                	jne    4069d8 <__sprintf_chk@plt+0x4148>
  4069c6:	49 89 d4             	mov    %rdx,%r12
  4069c9:	45 31 f6             	xor    %r14d,%r14d
  4069cc:	e9 2e fd ff ff       	jmpq   4066ff <__sprintf_chk@plt+0x3e6f>
  4069d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4069d8:	80 3d b9 47 21 00 00 	cmpb   $0x0,0x2147b9(%rip)        # 61b198 <stderr@@GLIBC_2.2.5+0xb48>
  4069df:	74 27                	je     406a08 <__sprintf_chk@plt+0x4178>
  4069e1:	80 bd b1 00 00 00 00 	cmpb   $0x0,0xb1(%rbp)
  4069e8:	74 36                	je     406a20 <__sprintf_chk@plt+0x4190>
  4069ea:	44 8b bd a4 00 00 00 	mov    0xa4(%rbp),%r15d
  4069f1:	41 be 01 00 00 00    	mov    $0x1,%r14d
  4069f7:	41 83 f6 01          	xor    $0x1,%r14d
  4069fb:	49 89 d4             	mov    %rdx,%r12
  4069fe:	e9 ad fb ff ff       	jmpq   4065b0 <__sprintf_chk@plt+0x3d20>
  406a03:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406a08:	44 0f b6 b5 b1 00 00 	movzbl 0xb1(%rbp),%r14d
  406a0f:	00 
  406a10:	44 8b 7d 28          	mov    0x28(%rbp),%r15d
  406a14:	eb e1                	jmp    4069f7 <__sprintf_chk@plt+0x4167>
  406a16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  406a1d:	00 00 00 
  406a20:	45 31 f6             	xor    %r14d,%r14d
  406a23:	eb eb                	jmp    406a10 <__sprintf_chk@plt+0x4180>
  406a25:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  406a2c:	00 00 00 00 
  406a30:	80 3d f2 46 21 00 00 	cmpb   $0x0,0x2146f2(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  406a37:	74 16                	je     406a4f <__sprintf_chk@plt+0x41bf>
  406a39:	48 83 ec 08          	sub    $0x8,%rsp
  406a3d:	bf 04 00 00 00       	mov    $0x4,%edi
  406a42:	e8 89 e2 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  406a47:	84 c0                	test   %al,%al
  406a49:	75 0d                	jne    406a58 <__sprintf_chk@plt+0x41c8>
  406a4b:	48 83 c4 08          	add    $0x8,%rsp
  406a4f:	f3 c3                	repz retq 
  406a51:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406a58:	bf e0 a3 61 00       	mov    $0x61a3e0,%edi
  406a5d:	e8 de f9 ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406a62:	bf 20 a4 61 00       	mov    $0x61a420,%edi
  406a67:	e8 d4 f9 ff ff       	callq  406440 <__sprintf_chk@plt+0x3bb0>
  406a6c:	bf f0 a3 61 00       	mov    $0x61a3f0,%edi
  406a71:	48 83 c4 08          	add    $0x8,%rsp
  406a75:	e9 c6 f9 ff ff       	jmpq   406440 <__sprintf_chk@plt+0x3bb0>
  406a7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  406a80:	41 56                	push   %r14
  406a82:	41 55                	push   %r13
  406a84:	41 89 cd             	mov    %ecx,%r13d
  406a87:	41 54                	push   %r12
  406a89:	49 89 fc             	mov    %rdi,%r12
  406a8c:	55                   	push   %rbp
  406a8d:	48 89 d5             	mov    %rdx,%rbp
  406a90:	53                   	push   %rbx
  406a91:	48 89 f3             	mov    %rsi,%rbx
  406a94:	48 81 ec 10 01 00 00 	sub    $0x110,%rsp
  406a9b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  406aa2:	00 00 
  406aa4:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
  406aab:	00 
  406aac:	31 c0                	xor    %eax,%eax
  406aae:	48 83 3d 92 3c 21 00 	cmpq   $0x0,0x213c92(%rip)        # 61a748 <stderr@@GLIBC_2.2.5+0xf8>
  406ab5:	00 
  406ab6:	74 23                	je     406adb <__sprintf_chk@plt+0x424b>
  406ab8:	be 66 37 41 00       	mov    $0x413766,%esi
  406abd:	48 89 df             	mov    %rbx,%rdi
  406ac0:	e8 9b bd ff ff       	callq  402860 <strstr@plt>
  406ac5:	48 85 c0             	test   %rax,%rax
  406ac8:	49 89 c6             	mov    %rax,%r14
  406acb:	74 0e                	je     406adb <__sprintf_chk@plt+0x424b>
  406acd:	48 89 df             	mov    %rbx,%rdi
  406ad0:	e8 ab b8 ff ff       	callq  402380 <strlen@plt>
  406ad5:	48 83 f8 65          	cmp    $0x65,%rax
  406ad9:	76 45                	jbe    406b20 <__sprintf_chk@plt+0x4290>
  406adb:	45 31 c0             	xor    %r8d,%r8d
  406ade:	48 89 e9             	mov    %rbp,%rcx
  406ae1:	45 89 e9             	mov    %r13d,%r9d
  406ae4:	48 89 da             	mov    %rbx,%rdx
  406ae7:	be e9 03 00 00       	mov    $0x3e9,%esi
  406aec:	4c 89 e7             	mov    %r12,%rdi
  406aef:	e8 0c 9b 00 00       	callq  410600 <__sprintf_chk@plt+0xdd70>
  406af4:	48 8b 8c 24 08 01 00 	mov    0x108(%rsp),%rcx
  406afb:	00 
  406afc:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  406b03:	00 00 
  406b05:	75 5f                	jne    406b66 <__sprintf_chk@plt+0x42d6>
  406b07:	48 81 c4 10 01 00 00 	add    $0x110,%rsp
  406b0e:	5b                   	pop    %rbx
  406b0f:	5d                   	pop    %rbp
  406b10:	41 5c                	pop    %r12
  406b12:	41 5d                	pop    %r13
  406b14:	41 5e                	pop    %r14
  406b16:	c3                   	retq   
  406b17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  406b1e:	00 00 
  406b20:	4c 89 f2             	mov    %r14,%rdx
  406b23:	48 89 de             	mov    %rbx,%rsi
  406b26:	b9 05 01 00 00       	mov    $0x105,%ecx
  406b2b:	48 29 da             	sub    %rbx,%rdx
  406b2e:	48 89 e7             	mov    %rsp,%rdi
  406b31:	48 89 e3             	mov    %rsp,%rbx
  406b34:	e8 d7 b6 ff ff       	callq  402210 <__mempcpy_chk@plt>
  406b39:	48 63 4d 10          	movslq 0x10(%rbp),%rcx
  406b3d:	48 89 c7             	mov    %rax,%rdi
  406b40:	48 8d 14 89          	lea    (%rcx,%rcx,4),%rdx
  406b44:	48 c1 e2 05          	shl    $0x5,%rdx
  406b48:	48 8d b4 11 60 a7 61 	lea    0x61a760(%rcx,%rdx,1),%rsi
  406b4f:	00 
  406b50:	e8 fb b7 ff ff       	callq  402350 <stpcpy@plt>
  406b55:	49 8d 76 02          	lea    0x2(%r14),%rsi
  406b59:	48 89 c7             	mov    %rax,%rdi
  406b5c:	e8 ff b6 ff ff       	callq  402260 <strcpy@plt>
  406b61:	e9 75 ff ff ff       	jmpq   406adb <__sprintf_chk@plt+0x424b>
  406b66:	e8 35 b8 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  406b6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406b70:	41 57                	push   %r15
  406b72:	41 56                	push   %r14
  406b74:	41 55                	push   %r13
  406b76:	41 54                	push   %r12
  406b78:	55                   	push   %rbp
  406b79:	53                   	push   %rbx
  406b7a:	48 89 fb             	mov    %rdi,%rbx
  406b7d:	48 81 ec b8 12 00 00 	sub    $0x12b8,%rsp
  406b84:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  406b8b:	00 00 
  406b8d:	48 89 84 24 a8 12 00 	mov    %rax,0x12a8(%rsp)
  406b94:	00 
  406b95:	31 c0                	xor    %eax,%eax
  406b97:	80 bf b0 00 00 00 00 	cmpb   $0x0,0xb0(%rdi)
  406b9e:	0f 84 5c 02 00 00    	je     406e00 <__sprintf_chk@plt+0x4570>
  406ba4:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
  406ba9:	48 8d 7f 10          	lea    0x10(%rdi),%rdi
  406bad:	4c 89 ee             	mov    %r13,%rsi
  406bb0:	e8 4b 3a 00 00       	callq  40a600 <__sprintf_chk@plt+0x7d70>
  406bb5:	80 3d c0 45 21 00 00 	cmpb   $0x0,0x2145c0(%rip)        # 61b17c <stderr@@GLIBC_2.2.5+0xb2c>
  406bbc:	0f 85 7f 02 00 00    	jne    406e41 <__sprintf_chk@plt+0x45b1>
  406bc2:	c6 44 24 4a 00       	movb   $0x0,0x4a(%rsp)
  406bc7:	8b 05 7f 45 21 00    	mov    0x21457f(%rip),%eax        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  406bcd:	83 f8 01             	cmp    $0x1,%eax
  406bd0:	0f 84 9a 02 00 00    	je     406e70 <__sprintf_chk@plt+0x45e0>
  406bd6:	0f 82 64 04 00 00    	jb     407040 <__sprintf_chk@plt+0x47b0>
  406bdc:	83 f8 02             	cmp    $0x2,%eax
  406bdf:	0f 85 ab 02 00 00    	jne    406e90 <__sprintf_chk@plt+0x4600>
  406be5:	48 8b 43 60          	mov    0x60(%rbx),%rax
  406be9:	48 8b 53 58          	mov    0x58(%rbx),%rdx
  406bed:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  406bf2:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  406bf7:	80 3d 16 45 21 00 00 	cmpb   $0x0,0x214516(%rip)        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  406bfe:	4c 8d a4 24 60 04 00 	lea    0x460(%rsp),%r12
  406c05:	00 
  406c06:	4c 89 e5             	mov    %r12,%rbp
  406c09:	0f 85 51 04 00 00    	jne    407060 <__sprintf_chk@plt+0x47d0>
  406c0f:	80 3d 2e 45 21 00 00 	cmpb   $0x0,0x21452e(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  406c16:	74 77                	je     406c8f <__sprintf_chk@plt+0x43ff>
  406c18:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  406c1f:	41 be 64 37 41 00    	mov    $0x413764,%r14d
  406c25:	0f 85 5d 06 00 00    	jne    407288 <__sprintf_chk@plt+0x49f8>
  406c2b:	44 8b 3d 42 45 21 00 	mov    0x214542(%rip),%r15d        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  406c32:	31 f6                	xor    %esi,%esi
  406c34:	4c 89 f7             	mov    %r14,%rdi
  406c37:	e8 e4 67 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  406c3c:	41 29 c7             	sub    %eax,%r15d
  406c3f:	45 85 ff             	test   %r15d,%r15d
  406c42:	44 89 f8             	mov    %r15d,%eax
  406c45:	0f 8e 05 03 00 00    	jle    406f50 <__sprintf_chk@plt+0x46c0>
  406c4b:	83 e8 01             	sub    $0x1,%eax
  406c4e:	89 c2                	mov    %eax,%edx
  406c50:	48 8d 4c 15 01       	lea    0x1(%rbp,%rdx,1),%rcx
  406c55:	48 89 ea             	mov    %rbp,%rdx
  406c58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  406c5f:	00 
  406c60:	48 83 c2 01          	add    $0x1,%rdx
  406c64:	c6 42 ff 20          	movb   $0x20,-0x1(%rdx)
  406c68:	48 39 ca             	cmp    %rcx,%rdx
  406c6b:	75 f3                	jne    406c60 <__sprintf_chk@plt+0x43d0>
  406c6d:	48 98                	cltq   
  406c6f:	48 8d 54 05 01       	lea    0x1(%rbp,%rax,1),%rdx
  406c74:	49 83 c6 01          	add    $0x1,%r14
  406c78:	41 0f b6 46 ff       	movzbl -0x1(%r14),%eax
  406c7d:	48 8d 6a 01          	lea    0x1(%rdx),%rbp
  406c81:	84 c0                	test   %al,%al
  406c83:	88 45 ff             	mov    %al,-0x1(%rbp)
  406c86:	0f 85 c4 02 00 00    	jne    406f50 <__sprintf_chk@plt+0x46c0>
  406c8c:	c6 02 20             	movb   $0x20,(%rdx)
  406c8f:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  406c96:	b8 64 37 41 00       	mov    $0x413764,%eax
  406c9b:	0f 85 7f 04 00 00    	jne    407120 <__sprintf_chk@plt+0x4890>
  406ca1:	44 8b 0d c8 44 21 00 	mov    0x2144c8(%rip),%r9d        # 61b170 <stderr@@GLIBC_2.2.5+0xb20>
  406ca8:	48 89 ef             	mov    %rbp,%rdi
  406cab:	48 89 04 24          	mov    %rax,(%rsp)
  406caf:	4d 89 e8             	mov    %r13,%r8
  406cb2:	b9 69 37 41 00       	mov    $0x413769,%ecx
  406cb7:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  406cbe:	be 01 00 00 00       	mov    $0x1,%esi
  406cc3:	31 c0                	xor    %eax,%eax
  406cc5:	e8 c6 bb ff ff       	callq  402890 <__sprintf_chk@plt>
  406cca:	48 89 ef             	mov    %rbp,%rdi
  406ccd:	e8 ae b6 ff ff       	callq  402380 <strlen@plt>
  406cd2:	48 01 c5             	add    %rax,%rbp
  406cd5:	80 3d 54 44 21 00 00 	cmpb   $0x0,0x214454(%rip)        # 61b130 <stderr@@GLIBC_2.2.5+0xae0>
  406cdc:	0f 85 0e 04 00 00    	jne    4070f0 <__sprintf_chk@plt+0x4860>
  406ce2:	80 3d 80 38 21 00 00 	cmpb   $0x0,0x213880(%rip)        # 61a569 <_fini@@Base+0x20866d>
  406ce9:	75 1d                	jne    406d08 <__sprintf_chk@plt+0x4478>
  406ceb:	80 3d 76 38 21 00 00 	cmpb   $0x0,0x213876(%rip)        # 61a568 <_fini@@Base+0x20866c>
  406cf2:	75 14                	jne    406d08 <__sprintf_chk@plt+0x4478>
  406cf4:	80 3d 4b 44 21 00 00 	cmpb   $0x0,0x21444b(%rip)        # 61b146 <stderr@@GLIBC_2.2.5+0xaf6>
  406cfb:	0f 84 47 04 00 00    	je     407148 <__sprintf_chk@plt+0x48b8>
  406d01:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406d08:	48 8b 35 01 39 21 00 	mov    0x213901(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  406d0f:	4c 89 e7             	mov    %r12,%rdi
  406d12:	4c 29 e5             	sub    %r12,%rbp
  406d15:	e8 06 b8 ff ff       	callq  402520 <fputs_unlocked@plt>
  406d1a:	48 01 2d f7 42 21 00 	add    %rbp,0x2142f7(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  406d21:	80 3d 41 38 21 00 00 	cmpb   $0x0,0x213841(%rip)        # 61a569 <_fini@@Base+0x20866d>
  406d28:	0f 85 1a 05 00 00    	jne    407248 <__sprintf_chk@plt+0x49b8>
  406d2e:	80 3d 33 38 21 00 00 	cmpb   $0x0,0x213833(%rip)        # 61a568 <_fini@@Base+0x20866c>
  406d35:	0f 85 d5 04 00 00    	jne    407210 <__sprintf_chk@plt+0x4980>
  406d3b:	80 3d 04 44 21 00 00 	cmpb   $0x0,0x214404(%rip)        # 61b146 <stderr@@GLIBC_2.2.5+0xaf6>
  406d42:	0f 85 88 04 00 00    	jne    4071d0 <__sprintf_chk@plt+0x4940>
  406d48:	80 3d 2e 44 21 00 00 	cmpb   $0x0,0x21442e(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  406d4f:	4c 89 e5             	mov    %r12,%rbp
  406d52:	0f 85 9d 04 00 00    	jne    4071f5 <__sprintf_chk@plt+0x4965>
  406d58:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  406d5f:	0f 84 fb 01 00 00    	je     406f60 <__sprintf_chk@plt+0x46d0>
  406d65:	8b 43 28             	mov    0x28(%rbx),%eax
  406d68:	25 00 b0 00 00       	and    $0xb000,%eax
  406d6d:	3d 00 20 00 00       	cmp    $0x2000,%eax
  406d72:	0f 84 38 05 00 00    	je     4072b0 <__sprintf_chk@plt+0x4a20>
  406d78:	48 8b 7b 40          	mov    0x40(%rbx),%rdi
  406d7c:	4c 8b 05 dd 37 21 00 	mov    0x2137dd(%rip),%r8        # 61a560 <_fini@@Base+0x208664>
  406d83:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  406d88:	8b 15 a6 43 21 00    	mov    0x2143a6(%rip),%edx        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  406d8e:	b9 01 00 00 00       	mov    $0x1,%ecx
  406d93:	e8 d8 4f 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  406d98:	49 89 c6             	mov    %rax,%r14
  406d9b:	44 8b 2d b2 43 21 00 	mov    0x2143b2(%rip),%r13d        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  406da2:	31 f6                	xor    %esi,%esi
  406da4:	4c 89 f7             	mov    %r14,%rdi
  406da7:	e8 74 66 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  406dac:	41 29 c5             	sub    %eax,%r13d
  406daf:	45 85 ed             	test   %r13d,%r13d
  406db2:	44 89 e8             	mov    %r13d,%eax
  406db5:	7e 25                	jle    406ddc <__sprintf_chk@plt+0x454c>
  406db7:	83 e8 01             	sub    $0x1,%eax
  406dba:	89 c2                	mov    %eax,%edx
  406dbc:	48 8d 4c 15 01       	lea    0x1(%rbp,%rdx,1),%rcx
  406dc1:	48 89 ea             	mov    %rbp,%rdx
  406dc4:	0f 1f 40 00          	nopl   0x0(%rax)
  406dc8:	48 83 c2 01          	add    $0x1,%rdx
  406dcc:	c6 42 ff 20          	movb   $0x20,-0x1(%rdx)
  406dd0:	48 39 ca             	cmp    %rcx,%rdx
  406dd3:	75 f3                	jne    406dc8 <__sprintf_chk@plt+0x4538>
  406dd5:	48 98                	cltq   
  406dd7:	48 8d 6c 05 01       	lea    0x1(%rbp,%rax,1),%rbp
  406ddc:	49 83 c6 01          	add    $0x1,%r14
  406de0:	41 0f b6 46 ff       	movzbl -0x1(%r14),%eax
  406de5:	4c 8d 6d 01          	lea    0x1(%rbp),%r13
  406de9:	84 c0                	test   %al,%al
  406deb:	41 88 45 ff          	mov    %al,-0x1(%r13)
  406def:	0f 84 a3 00 00 00    	je     406e98 <__sprintf_chk@plt+0x4608>
  406df5:	4c 89 ed             	mov    %r13,%rbp
  406df8:	eb e2                	jmp    406ddc <__sprintf_chk@plt+0x454c>
  406dfa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  406e00:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  406e06:	80 3d 6f 43 21 00 00 	cmpb   $0x0,0x21436f(%rip)        # 61b17c <stderr@@GLIBC_2.2.5+0xb2c>
  406e0d:	4c 8d 6c 24 40       	lea    0x40(%rsp),%r13
  406e12:	b9 3f 3f 00 00       	mov    $0x3f3f,%ecx
  406e17:	0f b6 80 28 37 41 00 	movzbl 0x413728(%rax),%eax
  406e1e:	88 44 24 40          	mov    %al,0x40(%rsp)
  406e22:	48 b8 3f 3f 3f 3f 3f 	movabs $0x3f3f3f3f3f3f3f3f,%rax
  406e29:	3f 3f 3f 
  406e2c:	48 89 44 24 41       	mov    %rax,0x41(%rsp)
  406e31:	66 41 89 4d 09       	mov    %cx,0x9(%r13)
  406e36:	c6 44 24 4b 00       	movb   $0x0,0x4b(%rsp)
  406e3b:	0f 84 81 fd ff ff    	je     406bc2 <__sprintf_chk@plt+0x4332>
  406e41:	8b 83 b4 00 00 00    	mov    0xb4(%rbx),%eax
  406e47:	83 f8 01             	cmp    $0x1,%eax
  406e4a:	0f 84 e8 02 00 00    	je     407138 <__sprintf_chk@plt+0x48a8>
  406e50:	83 f8 02             	cmp    $0x2,%eax
  406e53:	0f 85 6e fd ff ff    	jne    406bc7 <__sprintf_chk@plt+0x4337>
  406e59:	8b 05 ed 42 21 00    	mov    0x2142ed(%rip),%eax        # 61b14c <stderr@@GLIBC_2.2.5+0xafc>
  406e5f:	c6 44 24 4a 2b       	movb   $0x2b,0x4a(%rsp)
  406e64:	83 f8 01             	cmp    $0x1,%eax
  406e67:	0f 85 69 fd ff ff    	jne    406bd6 <__sprintf_chk@plt+0x4346>
  406e6d:	0f 1f 00             	nopl   (%rax)
  406e70:	48 8b 83 80 00 00 00 	mov    0x80(%rbx),%rax
  406e77:	48 8b 53 78          	mov    0x78(%rbx),%rdx
  406e7b:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  406e80:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  406e85:	e9 6d fd ff ff       	jmpq   406bf7 <__sprintf_chk@plt+0x4367>
  406e8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  406e90:	e8 8b b3 ff ff       	callq  402220 <abort@plt>
  406e95:	0f 1f 00             	nopl   (%rax)
  406e98:	c6 45 00 20          	movb   $0x20,0x0(%rbp)
  406e9c:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  406ea1:	e8 5a b3 ff ff       	callq  402200 <localtime@plt>
  406ea6:	41 c6 45 00 01       	movb   $0x1,0x0(%r13)
  406eab:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  406eb2:	0f 84 c8 00 00 00    	je     406f80 <__sprintf_chk@plt+0x46f0>
  406eb8:	48 85 c0             	test   %rax,%rax
  406ebb:	0f 84 a3 04 00 00    	je     407364 <__sprintf_chk@plt+0x4ad4>
  406ec1:	48 8b 15 b8 42 21 00 	mov    0x2142b8(%rip),%rdx        # 61b180 <stderr@@GLIBC_2.2.5+0xb30>
  406ec8:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  406ecd:	48 8b 3d b4 42 21 00 	mov    0x2142b4(%rip),%rdi        # 61b188 <stderr@@GLIBC_2.2.5+0xb38>
  406ed4:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
  406ed9:	48 39 d6             	cmp    %rdx,%rsi
  406edc:	0f 8f 9e 04 00 00    	jg     407380 <__sprintf_chk@plt+0x4af0>
  406ee2:	7c 08                	jl     406eec <__sprintf_chk@plt+0x465c>
  406ee4:	39 cf                	cmp    %ecx,%edi
  406ee6:	0f 88 94 04 00 00    	js     407380 <__sprintf_chk@plt+0x4af0>
  406eec:	49 89 f8             	mov    %rdi,%r8
  406eef:	48 8d ba 54 3d 0f ff 	lea    -0xf0c2ac(%rdx),%rdi
  406ef6:	48 39 f7             	cmp    %rsi,%rdi
  406ef9:	0f 8d 69 03 00 00    	jge    407268 <__sprintf_chk@plt+0x49d8>
  406eff:	48 39 f2             	cmp    %rsi,%rdx
  406f02:	bf 01 00 00 00       	mov    $0x1,%edi
  406f07:	7f 17                	jg     406f20 <__sprintf_chk@plt+0x4690>
  406f09:	40 b7 00             	mov    $0x0,%dil
  406f0c:	7c 12                	jl     406f20 <__sprintf_chk@plt+0x4690>
  406f0e:	89 cf                	mov    %ecx,%edi
  406f10:	44 29 c7             	sub    %r8d,%edi
  406f13:	c1 ef 1f             	shr    $0x1f,%edi
  406f16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  406f1d:	00 00 00 
  406f20:	48 63 ff             	movslq %edi,%rdi
  406f23:	48 89 c2             	mov    %rax,%rdx
  406f26:	48 8b 34 fd d0 a3 61 	mov    0x61a3d0(,%rdi,8),%rsi
  406f2d:	00 
  406f2e:	4c 89 ef             	mov    %r13,%rdi
  406f31:	e8 4a fb ff ff       	callq  406a80 <__sprintf_chk@plt+0x41f0>
  406f36:	48 85 c0             	test   %rax,%rax
  406f39:	74 30                	je     406f6b <__sprintf_chk@plt+0x46db>
  406f3b:	4c 01 e8             	add    %r13,%rax
  406f3e:	4c 8d 68 01          	lea    0x1(%rax),%r13
  406f42:	c6 00 20             	movb   $0x20,(%rax)
  406f45:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  406f49:	eb 71                	jmp    406fbc <__sprintf_chk@plt+0x472c>
  406f4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406f50:	48 89 ea             	mov    %rbp,%rdx
  406f53:	e9 1c fd ff ff       	jmpq   406c74 <__sprintf_chk@plt+0x43e4>
  406f58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  406f5f:	00 
  406f60:	41 be 64 37 41 00    	mov    $0x413764,%r14d
  406f66:	e9 30 fe ff ff       	jmpq   406d9b <__sprintf_chk@plt+0x450b>
  406f6b:	41 80 7d 00 00       	cmpb   $0x0,0x0(%r13)
  406f70:	74 c9                	je     406f3b <__sprintf_chk@plt+0x46ab>
  406f72:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  406f79:	0f 85 e5 03 00 00    	jne    407364 <__sprintf_chk@plt+0x4ad4>
  406f7f:	90                   	nop
  406f80:	41 b9 64 37 41 00    	mov    $0x413764,%r9d
  406f86:	44 8b 05 37 34 21 00 	mov    0x213437(%rip),%r8d        # 61a3c4 <_fini@@Base+0x2084c8>
  406f8d:	45 85 c0             	test   %r8d,%r8d
  406f90:	0f 88 1b 04 00 00    	js     4073b1 <__sprintf_chk@plt+0x4b21>
  406f96:	4c 89 ef             	mov    %r13,%rdi
  406f99:	b9 79 37 41 00       	mov    $0x413779,%ecx
  406f9e:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  406fa5:	be 01 00 00 00       	mov    $0x1,%esi
  406faa:	31 c0                	xor    %eax,%eax
  406fac:	e8 df b8 ff ff       	callq  402890 <__sprintf_chk@plt>
  406fb1:	4c 89 ef             	mov    %r13,%rdi
  406fb4:	e8 c7 b3 ff ff       	callq  402380 <strlen@plt>
  406fb9:	49 01 c5             	add    %rax,%r13
  406fbc:	48 8b 35 4d 36 21 00 	mov    0x21364d(%rip),%rsi        # 61a610 <stdout@@GLIBC_2.2.5>
  406fc3:	4d 29 e5             	sub    %r12,%r13
  406fc6:	4c 89 e7             	mov    %r12,%rdi
  406fc9:	e8 52 b5 ff ff       	callq  402520 <fputs_unlocked@plt>
  406fce:	ba c0 af 61 00       	mov    $0x61afc0,%edx
  406fd3:	31 f6                	xor    %esi,%esi
  406fd5:	4c 89 e9             	mov    %r13,%rcx
  406fd8:	48 89 df             	mov    %rbx,%rdi
  406fdb:	4c 01 2d 36 40 21 00 	add    %r13,0x214036(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  406fe2:	e8 59 f5 ff ff       	callq  406540 <__sprintf_chk@plt+0x3cb0>
  406fe7:	8b 93 a0 00 00 00    	mov    0xa0(%rbx),%edx
  406fed:	48 89 c5             	mov    %rax,%rbp
  406ff0:	83 fa 06             	cmp    $0x6,%edx
  406ff3:	0f 84 67 01 00 00    	je     407160 <__sprintf_chk@plt+0x48d0>
  406ff9:	8b 05 2d 41 21 00    	mov    0x21412d(%rip),%eax        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  406fff:	85 c0                	test   %eax,%eax
  407001:	74 0f                	je     407012 <__sprintf_chk@plt+0x4782>
  407003:	0f b6 bb b0 00 00 00 	movzbl 0xb0(%rbx),%edi
  40700a:	8b 73 28             	mov    0x28(%rbx),%esi
  40700d:	e8 ee ec ff ff       	callq  405d00 <__sprintf_chk@plt+0x3470>
  407012:	48 8b 84 24 a8 12 00 	mov    0x12a8(%rsp),%rax
  407019:	00 
  40701a:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  407021:	00 00 
  407023:	0f 85 31 04 00 00    	jne    40745a <__sprintf_chk@plt+0x4bca>
  407029:	48 81 c4 b8 12 00 00 	add    $0x12b8,%rsp
  407030:	5b                   	pop    %rbx
  407031:	5d                   	pop    %rbp
  407032:	41 5c                	pop    %r12
  407034:	41 5d                	pop    %r13
  407036:	41 5e                	pop    %r14
  407038:	41 5f                	pop    %r15
  40703a:	c3                   	retq   
  40703b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407040:	48 8b 43 70          	mov    0x70(%rbx),%rax
  407044:	48 8b 53 68          	mov    0x68(%rbx),%rdx
  407048:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  40704d:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  407052:	e9 a0 fb ff ff       	jmpq   406bf7 <__sprintf_chk@plt+0x4367>
  407057:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40705e:	00 00 
  407060:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  407067:	41 b9 64 37 41 00    	mov    $0x413764,%r9d
  40706d:	74 16                	je     407085 <__sprintf_chk@plt+0x47f5>
  40706f:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
  407073:	48 85 ff             	test   %rdi,%rdi
  407076:	74 0d                	je     407085 <__sprintf_chk@plt+0x47f5>
  407078:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  40707d:	e8 ee 5c 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  407082:	49 89 c1             	mov    %rax,%r9
  407085:	4c 8d a4 24 60 04 00 	lea    0x460(%rsp),%r12
  40708c:	00 
  40708d:	44 8b 05 e4 40 21 00 	mov    0x2140e4(%rip),%r8d        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  407094:	ba 3b 0e 00 00       	mov    $0xe3b,%edx
  407099:	b9 79 37 41 00       	mov    $0x413779,%ecx
  40709e:	be 01 00 00 00       	mov    $0x1,%esi
  4070a3:	31 c0                	xor    %eax,%eax
  4070a5:	4c 89 e7             	mov    %r12,%rdi
  4070a8:	e8 e3 b7 ff ff       	callq  402890 <__sprintf_chk@plt>
  4070ad:	4c 89 e2             	mov    %r12,%rdx
  4070b0:	8b 0a                	mov    (%rdx),%ecx
  4070b2:	48 83 c2 04          	add    $0x4,%rdx
  4070b6:	8d 81 ff fe fe fe    	lea    -0x1010101(%rcx),%eax
  4070bc:	f7 d1                	not    %ecx
  4070be:	21 c8                	and    %ecx,%eax
  4070c0:	25 80 80 80 80       	and    $0x80808080,%eax
  4070c5:	74 e9                	je     4070b0 <__sprintf_chk@plt+0x4820>
  4070c7:	89 c1                	mov    %eax,%ecx
  4070c9:	48 8d 6a 02          	lea    0x2(%rdx),%rbp
  4070cd:	c1 e9 10             	shr    $0x10,%ecx
  4070d0:	a9 80 80 00 00       	test   $0x8080,%eax
  4070d5:	0f 44 c1             	cmove  %ecx,%eax
  4070d8:	48 0f 45 ea          	cmovne %rdx,%rbp
  4070dc:	00 c0                	add    %al,%al
  4070de:	48 83 dd 03          	sbb    $0x3,%rbp
  4070e2:	e9 28 fb ff ff       	jmpq   406c0f <__sprintf_chk@plt+0x437f>
  4070e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4070ee:	00 00 
  4070f0:	48 8b 0d 19 35 21 00 	mov    0x213519(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  4070f7:	ba 02 00 00 00       	mov    $0x2,%edx
  4070fc:	be 01 00 00 00       	mov    $0x1,%esi
  407101:	bf 71 37 41 00       	mov    $0x413771,%edi
  407106:	e8 b5 b5 ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  40710b:	48 83 05 05 3f 21 00 	addq   $0x2,0x213f05(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  407112:	02 
  407113:	e9 ca fb ff ff       	jmpq   406ce2 <__sprintf_chk@plt+0x4452>
  407118:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40711f:	00 
  407120:	48 8b 7b 20          	mov    0x20(%rbx),%rdi
  407124:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  407129:	e8 42 5c 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  40712e:	e9 6e fb ff ff       	jmpq   406ca1 <__sprintf_chk@plt+0x4411>
  407133:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407138:	c6 44 24 4a 2e       	movb   $0x2e,0x4a(%rsp)
  40713d:	e9 85 fa ff ff       	jmpq   406bc7 <__sprintf_chk@plt+0x4337>
  407142:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407148:	80 3d 2e 40 21 00 00 	cmpb   $0x0,0x21402e(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  40714f:	0f 84 03 fc ff ff    	je     406d58 <__sprintf_chk@plt+0x44c8>
  407155:	e9 ae fb ff ff       	jmpq   406d08 <__sprintf_chk@plt+0x4478>
  40715a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407160:	48 83 7b 08 00       	cmpq   $0x0,0x8(%rbx)
  407165:	0f 84 a7 fe ff ff    	je     407012 <__sprintf_chk@plt+0x4782>
  40716b:	48 8b 0d 9e 34 21 00 	mov    0x21349e(%rip),%rcx        # 61a610 <stdout@@GLIBC_2.2.5>
  407172:	ba 04 00 00 00       	mov    $0x4,%edx
  407177:	be 01 00 00 00       	mov    $0x1,%esi
  40717c:	bf 7e 37 41 00       	mov    $0x41377e,%edi
  407181:	e8 3a b5 ff ff       	callq  4026c0 <fwrite_unlocked@plt>
  407186:	49 8d 4c 2d 04       	lea    0x4(%r13,%rbp,1),%rcx
  40718b:	31 d2                	xor    %edx,%edx
  40718d:	be 01 00 00 00       	mov    $0x1,%esi
  407192:	48 89 df             	mov    %rbx,%rdi
  407195:	48 83 05 7b 3e 21 00 	addq   $0x4,0x213e7b(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  40719c:	04 
  40719d:	e8 9e f3 ff ff       	callq  406540 <__sprintf_chk@plt+0x3cb0>
  4071a2:	8b 15 84 3f 21 00    	mov    0x213f84(%rip),%edx        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  4071a8:	85 d2                	test   %edx,%edx
  4071aa:	0f 84 62 fe ff ff    	je     407012 <__sprintf_chk@plt+0x4782>
  4071b0:	8b b3 a4 00 00 00    	mov    0xa4(%rbx),%esi
  4071b6:	31 d2                	xor    %edx,%edx
  4071b8:	bf 01 00 00 00       	mov    $0x1,%edi
  4071bd:	e8 3e eb ff ff       	callq  405d00 <__sprintf_chk@plt+0x3470>
  4071c2:	e9 4b fe ff ff       	jmpq   407012 <__sprintf_chk@plt+0x4782>
  4071c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4071ce:	00 00 
  4071d0:	0f b6 93 b0 00 00 00 	movzbl 0xb0(%rbx),%edx
  4071d7:	8b 7b 2c             	mov    0x2c(%rbx),%edi
  4071da:	4c 89 e5             	mov    %r12,%rbp
  4071dd:	8b 35 7d 3f 21 00    	mov    0x213f7d(%rip),%esi        # 61b160 <stderr@@GLIBC_2.2.5+0xb10>
  4071e3:	e8 c8 e5 ff ff       	callq  4057b0 <__sprintf_chk@plt+0x2f20>
  4071e8:	80 3d 8e 3f 21 00 00 	cmpb   $0x0,0x213f8e(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  4071ef:	0f 84 63 fb ff ff    	je     406d58 <__sprintf_chk@plt+0x44c8>
  4071f5:	48 8b bb a8 00 00 00 	mov    0xa8(%rbx),%rdi
  4071fc:	8b 15 6a 3f 21 00    	mov    0x213f6a(%rip),%edx        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  407202:	31 f6                	xor    %esi,%esi
  407204:	e8 f7 e4 ff ff       	callq  405700 <__sprintf_chk@plt+0x2e70>
  407209:	e9 4a fb ff ff       	jmpq   406d58 <__sprintf_chk@plt+0x44c8>
  40720e:	66 90                	xchg   %ax,%ax
  407210:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  407217:	8b 43 30             	mov    0x30(%rbx),%eax
  40721a:	bf 64 37 41 00       	mov    $0x413764,%edi
  40721f:	8b 15 3f 3f 21 00    	mov    0x213f3f(%rip),%edx        # 61b164 <stderr@@GLIBC_2.2.5+0xb14>
  407225:	89 c6                	mov    %eax,%esi
  407227:	74 0f                	je     407238 <__sprintf_chk@plt+0x49a8>
  407229:	31 ff                	xor    %edi,%edi
  40722b:	80 3d 13 3f 21 00 00 	cmpb   $0x0,0x213f13(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  407232:	0f 84 dd 01 00 00    	je     407415 <__sprintf_chk@plt+0x4b85>
  407238:	e8 c3 e4 ff ff       	callq  405700 <__sprintf_chk@plt+0x2e70>
  40723d:	e9 f9 fa ff ff       	jmpq   406d3b <__sprintf_chk@plt+0x44ab>
  407242:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407248:	0f b6 93 b0 00 00 00 	movzbl 0xb0(%rbx),%edx
  40724f:	8b 7b 2c             	mov    0x2c(%rbx),%edi
  407252:	8b 35 10 3f 21 00    	mov    0x213f10(%rip),%esi        # 61b168 <stderr@@GLIBC_2.2.5+0xb18>
  407258:	e8 53 e5 ff ff       	callq  4057b0 <__sprintf_chk@plt+0x2f20>
  40725d:	e9 cc fa ff ff       	jmpq   406d2e <__sprintf_chk@plt+0x449e>
  407262:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407268:	bf 00 00 00 00       	mov    $0x0,%edi
  40726d:	0f 8f ad fc ff ff    	jg     406f20 <__sprintf_chk@plt+0x4690>
  407273:	41 39 c8             	cmp    %ecx,%r8d
  407276:	0f 89 a4 fc ff ff    	jns    406f20 <__sprintf_chk@plt+0x4690>
  40727c:	e9 7e fc ff ff       	jmpq   406eff <__sprintf_chk@plt+0x466f>
  407281:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407288:	48 8b 7b 50          	mov    0x50(%rbx),%rdi
  40728c:	4c 8b 05 a5 3e 21 00 	mov    0x213ea5(%rip),%r8        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  407293:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  407298:	8b 15 a2 3e 21 00    	mov    0x213ea2(%rip),%edx        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  40729e:	b9 00 02 00 00       	mov    $0x200,%ecx
  4072a3:	e8 c8 4a 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  4072a8:	49 89 c6             	mov    %rax,%r14
  4072ab:	e9 7b f9 ff ff       	jmpq   406c2b <__sprintf_chk@plt+0x439b>
  4072b0:	48 8b 43 38          	mov    0x38(%rbx),%rax
  4072b4:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
  4072b9:	41 bd fe ff ff ff    	mov    $0xfffffffe,%r13d
  4072bf:	44 2b 2d 96 3e 21 00 	sub    0x213e96(%rip),%r13d        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  4072c6:	44 2b 2d 8b 3e 21 00 	sub    0x213e8b(%rip),%r13d        # 61b158 <stderr@@GLIBC_2.2.5+0xb08>
  4072cd:	48 89 c7             	mov    %rax,%rdi
  4072d0:	0f b6 c0             	movzbl %al,%eax
  4072d3:	44 03 2d 7a 3e 21 00 	add    0x213e7a(%rip),%r13d        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  4072da:	48 c1 ef 0c          	shr    $0xc,%rdi
  4072de:	40 80 e7 00          	and    $0x0,%dil
  4072e2:	09 c7                	or     %eax,%edi
  4072e4:	e8 87 5a 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  4072e9:	48 8b 53 38          	mov    0x38(%rbx),%rdx
  4072ed:	49 89 c7             	mov    %rax,%r15
  4072f0:	48 8d 74 24 50       	lea    0x50(%rsp),%rsi
  4072f5:	44 8b 35 5c 3e 21 00 	mov    0x213e5c(%rip),%r14d        # 61b158 <stderr@@GLIBC_2.2.5+0xb08>
  4072fc:	48 89 d7             	mov    %rdx,%rdi
  4072ff:	48 c1 ea 08          	shr    $0x8,%rdx
  407303:	89 d0                	mov    %edx,%eax
  407305:	48 c1 ef 20          	shr    $0x20,%rdi
  407309:	25 ff 0f 00 00       	and    $0xfff,%eax
  40730e:	81 e7 00 f0 ff ff    	and    $0xfffff000,%edi
  407314:	09 c7                	or     %eax,%edi
  407316:	e8 55 5a 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  40731b:	45 31 c0             	xor    %r8d,%r8d
  40731e:	45 85 ed             	test   %r13d,%r13d
  407321:	49 89 c1             	mov    %rax,%r9
  407324:	45 0f 49 c5          	cmovns %r13d,%r8d
  407328:	44 03 05 2d 3e 21 00 	add    0x213e2d(%rip),%r8d        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  40732f:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  407334:	44 89 34 24          	mov    %r14d,(%rsp)
  407338:	b9 74 37 41 00       	mov    $0x413774,%ecx
  40733d:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  407344:	be 01 00 00 00       	mov    $0x1,%esi
  407349:	48 89 ef             	mov    %rbp,%rdi
  40734c:	31 c0                	xor    %eax,%eax
  40734e:	e8 3d b5 ff ff       	callq  402890 <__sprintf_chk@plt>
  407353:	48 63 05 fa 3d 21 00 	movslq 0x213dfa(%rip),%rax        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  40735a:	4c 8d 6c 05 01       	lea    0x1(%rbp,%rax,1),%r13
  40735f:	e9 38 fb ff ff       	jmpq   406e9c <__sprintf_chk@plt+0x460c>
  407364:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  407369:	48 8d 74 24 50       	lea    0x50(%rsp),%rsi
  40736e:	e8 5d 59 00 00       	callq  40ccd0 <__sprintf_chk@plt+0xa440>
  407373:	49 89 c1             	mov    %rax,%r9
  407376:	e9 0b fc ff ff       	jmpq   406f86 <__sprintf_chk@plt+0x46f6>
  40737b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407380:	bf 80 b1 61 00       	mov    $0x61b180,%edi
  407385:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40738a:	e8 a1 37 00 00       	callq  40ab30 <__sprintf_chk@plt+0x82a0>
  40738f:	48 8b 15 ea 3d 21 00 	mov    0x213dea(%rip),%rdx        # 61b180 <stderr@@GLIBC_2.2.5+0xb30>
  407396:	4c 8b 05 eb 3d 21 00 	mov    0x213deb(%rip),%r8        # 61b188 <stderr@@GLIBC_2.2.5+0xb38>
  40739d:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  4073a2:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
  4073a7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  4073ac:	e9 3e fb ff ff       	jmpq   406eef <__sprintf_chk@plt+0x465f>
  4073b1:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
  4073b6:	4c 89 4c 24 10       	mov    %r9,0x10(%rsp)
  4073bb:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
  4073c2:	00 00 
  4073c4:	e8 37 ae ff ff       	callq  402200 <localtime@plt>
  4073c9:	48 85 c0             	test   %rax,%rax
  4073cc:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
  4073d1:	74 30                	je     407403 <__sprintf_chk@plt+0x4b73>
  4073d3:	48 8b 35 f6 2f 21 00 	mov    0x212ff6(%rip),%rsi        # 61a3d0 <_fini@@Base+0x2084d4>
  4073da:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  4073df:	31 c9                	xor    %ecx,%ecx
  4073e1:	48 89 c2             	mov    %rax,%rdx
  4073e4:	e8 97 f6 ff ff       	callq  406a80 <__sprintf_chk@plt+0x41f0>
  4073e9:	48 85 c0             	test   %rax,%rax
  4073ec:	44 8b 05 d1 2f 21 00 	mov    0x212fd1(%rip),%r8d        # 61a3c4 <_fini@@Base+0x2084c8>
  4073f3:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
  4073f8:	75 3c                	jne    407436 <__sprintf_chk@plt+0x4ba6>
  4073fa:	45 85 c0             	test   %r8d,%r8d
  4073fd:	0f 89 93 fb ff ff    	jns    406f96 <__sprintf_chk@plt+0x4706>
  407403:	c7 05 b7 2f 21 00 00 	movl   $0x0,0x212fb7(%rip)        # 61a3c4 <_fini@@Base+0x2084c8>
  40740a:	00 00 00 
  40740d:	45 31 c0             	xor    %r8d,%r8d
  407410:	e9 81 fb ff ff       	jmpq   406f96 <__sprintf_chk@plt+0x4706>
  407415:	89 c7                	mov    %eax,%edi
  407417:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  40741c:	89 54 24 10          	mov    %edx,0x10(%rsp)
  407420:	e8 1b 57 00 00       	callq  40cb40 <__sprintf_chk@plt+0xa2b0>
  407425:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  40742a:	48 89 c7             	mov    %rax,%rdi
  40742d:	8b 54 24 10          	mov    0x10(%rsp),%edx
  407431:	e9 02 fe ff ff       	jmpq   407238 <__sprintf_chk@plt+0x49a8>
  407436:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40743b:	31 d2                	xor    %edx,%edx
  40743d:	48 89 c6             	mov    %rax,%rsi
  407440:	4c 89 4c 24 10       	mov    %r9,0x10(%rsp)
  407445:	e8 f6 5d 00 00       	callq  40d240 <__sprintf_chk@plt+0xa9b0>
  40744a:	4c 8b 4c 24 10       	mov    0x10(%rsp),%r9
  40744f:	89 05 6f 2f 21 00    	mov    %eax,0x212f6f(%rip)        # 61a3c4 <_fini@@Base+0x2084c8>
  407455:	41 89 c0             	mov    %eax,%r8d
  407458:	eb a0                	jmp    4073fa <__sprintf_chk@plt+0x4b6a>
  40745a:	e8 41 af ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  40745f:	90                   	nop
  407460:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  407466:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40746d:	83 f8 09             	cmp    $0x9,%eax
  407470:	0f 94 c1             	sete   %cl
  407473:	83 f8 03             	cmp    $0x3,%eax
  407476:	0f 94 c2             	sete   %dl
  407479:	41 83 f8 09          	cmp    $0x9,%r8d
  40747d:	0f 94 c0             	sete   %al
  407480:	41 83 f8 03          	cmp    $0x3,%r8d
  407484:	41 0f 94 c0          	sete   %r8b
  407488:	44 09 c0             	or     %r8d,%eax
  40748b:	08 ca                	or     %cl,%dl
  40748d:	75 21                	jne    4074b0 <__sprintf_chk@plt+0x4c20>
  40748f:	84 d2                	test   %dl,%dl
  407491:	74 2d                	je     4074c0 <__sprintf_chk@plt+0x4c30>
  407493:	48 8b 4f 40          	mov    0x40(%rdi),%rcx
  407497:	48 39 4e 40          	cmp    %rcx,0x40(%rsi)
  40749b:	48 8b 06             	mov    (%rsi),%rax
  40749e:	48 8b 17             	mov    (%rdi),%rdx
  4074a1:	7f 15                	jg     4074b8 <__sprintf_chk@plt+0x4c28>
  4074a3:	7c 1f                	jl     4074c4 <__sprintf_chk@plt+0x4c34>
  4074a5:	48 89 d6             	mov    %rdx,%rsi
  4074a8:	48 89 c7             	mov    %rax,%rdi
  4074ab:	e9 a0 b0 ff ff       	jmpq   402550 <strcmp@plt>
  4074b0:	84 c0                	test   %al,%al
  4074b2:	75 db                	jne    40748f <__sprintf_chk@plt+0x4bff>
  4074b4:	0f 1f 40 00          	nopl   0x0(%rax)
  4074b8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4074bd:	c3                   	retq   
  4074be:	66 90                	xchg   %ax,%ax
  4074c0:	84 c0                	test   %al,%al
  4074c2:	74 cf                	je     407493 <__sprintf_chk@plt+0x4c03>
  4074c4:	b8 01 00 00 00       	mov    $0x1,%eax
  4074c9:	c3                   	retq   
  4074ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4074d0:	48 8b 4e 40          	mov    0x40(%rsi),%rcx
  4074d4:	48 39 4f 40          	cmp    %rcx,0x40(%rdi)
  4074d8:	48 8b 07             	mov    (%rdi),%rax
  4074db:	48 8b 16             	mov    (%rsi),%rdx
  4074de:	7f 10                	jg     4074f0 <__sprintf_chk@plt+0x4c60>
  4074e0:	7c 1e                	jl     407500 <__sprintf_chk@plt+0x4c70>
  4074e2:	48 89 d6             	mov    %rdx,%rsi
  4074e5:	48 89 c7             	mov    %rax,%rdi
  4074e8:	e9 33 db ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  4074ed:	0f 1f 00             	nopl   (%rax)
  4074f0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4074f5:	c3                   	retq   
  4074f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4074fd:	00 00 00 
  407500:	b8 01 00 00 00       	mov    $0x1,%eax
  407505:	c3                   	retq   
  407506:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40750d:	00 00 00 
  407510:	48 8b 4e 40          	mov    0x40(%rsi),%rcx
  407514:	48 39 4f 40          	cmp    %rcx,0x40(%rdi)
  407518:	48 8b 07             	mov    (%rdi),%rax
  40751b:	48 8b 16             	mov    (%rsi),%rdx
  40751e:	7f 10                	jg     407530 <__sprintf_chk@plt+0x4ca0>
  407520:	7c 1e                	jl     407540 <__sprintf_chk@plt+0x4cb0>
  407522:	48 89 d6             	mov    %rdx,%rsi
  407525:	48 89 c7             	mov    %rax,%rdi
  407528:	e9 23 b0 ff ff       	jmpq   402550 <strcmp@plt>
  40752d:	0f 1f 00             	nopl   (%rax)
  407530:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  407535:	c3                   	retq   
  407536:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40753d:	00 00 00 
  407540:	b8 01 00 00 00       	mov    $0x1,%eax
  407545:	c3                   	retq   
  407546:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40754d:	00 00 00 
  407550:	48 8b 4f 40          	mov    0x40(%rdi),%rcx
  407554:	48 39 4e 40          	cmp    %rcx,0x40(%rsi)
  407558:	48 8b 06             	mov    (%rsi),%rax
  40755b:	48 8b 17             	mov    (%rdi),%rdx
  40755e:	7f 10                	jg     407570 <__sprintf_chk@plt+0x4ce0>
  407560:	7c 1e                	jl     407580 <__sprintf_chk@plt+0x4cf0>
  407562:	48 89 d6             	mov    %rdx,%rsi
  407565:	48 89 c7             	mov    %rax,%rdi
  407568:	e9 b3 da ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  40756d:	0f 1f 00             	nopl   (%rax)
  407570:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  407575:	c3                   	retq   
  407576:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40757d:	00 00 00 
  407580:	b8 01 00 00 00       	mov    $0x1,%eax
  407585:	c3                   	retq   
  407586:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40758d:	00 00 00 
  407590:	48 8b 4f 40          	mov    0x40(%rdi),%rcx
  407594:	48 39 4e 40          	cmp    %rcx,0x40(%rsi)
  407598:	48 8b 06             	mov    (%rsi),%rax
  40759b:	48 8b 17             	mov    (%rdi),%rdx
  40759e:	7f 10                	jg     4075b0 <__sprintf_chk@plt+0x4d20>
  4075a0:	7c 1e                	jl     4075c0 <__sprintf_chk@plt+0x4d30>
  4075a2:	48 89 d6             	mov    %rdx,%rsi
  4075a5:	48 89 c7             	mov    %rax,%rdi
  4075a8:	e9 a3 af ff ff       	jmpq   402550 <strcmp@plt>
  4075ad:	0f 1f 00             	nopl   (%rax)
  4075b0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4075b5:	c3                   	retq   
  4075b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4075bd:	00 00 00 
  4075c0:	b8 01 00 00 00       	mov    $0x1,%eax
  4075c5:	c3                   	retq   
  4075c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4075cd:	00 00 00 
  4075d0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  4075d6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  4075dd:	83 f8 09             	cmp    $0x9,%eax
  4075e0:	0f 94 c1             	sete   %cl
  4075e3:	83 f8 03             	cmp    $0x3,%eax
  4075e6:	0f 94 c2             	sete   %dl
  4075e9:	41 83 f8 09          	cmp    $0x9,%r8d
  4075ed:	0f 94 c0             	sete   %al
  4075f0:	41 83 f8 03          	cmp    $0x3,%r8d
  4075f4:	41 0f 94 c0          	sete   %r8b
  4075f8:	44 09 c0             	or     %r8d,%eax
  4075fb:	08 ca                	or     %cl,%dl
  4075fd:	75 21                	jne    407620 <__sprintf_chk@plt+0x4d90>
  4075ff:	84 d2                	test   %dl,%dl
  407601:	74 2d                	je     407630 <__sprintf_chk@plt+0x4da0>
  407603:	48 8b 4e 40          	mov    0x40(%rsi),%rcx
  407607:	48 39 4f 40          	cmp    %rcx,0x40(%rdi)
  40760b:	48 8b 07             	mov    (%rdi),%rax
  40760e:	48 8b 16             	mov    (%rsi),%rdx
  407611:	7f 15                	jg     407628 <__sprintf_chk@plt+0x4d98>
  407613:	7c 1f                	jl     407634 <__sprintf_chk@plt+0x4da4>
  407615:	48 89 d6             	mov    %rdx,%rsi
  407618:	48 89 c7             	mov    %rax,%rdi
  40761b:	e9 00 da ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  407620:	84 c0                	test   %al,%al
  407622:	75 db                	jne    4075ff <__sprintf_chk@plt+0x4d6f>
  407624:	0f 1f 40 00          	nopl   0x0(%rax)
  407628:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40762d:	c3                   	retq   
  40762e:	66 90                	xchg   %ax,%ax
  407630:	84 c0                	test   %al,%al
  407632:	74 cf                	je     407603 <__sprintf_chk@plt+0x4d73>
  407634:	b8 01 00 00 00       	mov    $0x1,%eax
  407639:	c3                   	retq   
  40763a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407640:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  407646:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40764d:	83 f8 09             	cmp    $0x9,%eax
  407650:	0f 94 c1             	sete   %cl
  407653:	83 f8 03             	cmp    $0x3,%eax
  407656:	0f 94 c2             	sete   %dl
  407659:	41 83 f8 09          	cmp    $0x9,%r8d
  40765d:	0f 94 c0             	sete   %al
  407660:	41 83 f8 03          	cmp    $0x3,%r8d
  407664:	41 0f 94 c0          	sete   %r8b
  407668:	44 09 c0             	or     %r8d,%eax
  40766b:	08 ca                	or     %cl,%dl
  40766d:	75 21                	jne    407690 <__sprintf_chk@plt+0x4e00>
  40766f:	84 d2                	test   %dl,%dl
  407671:	74 2d                	je     4076a0 <__sprintf_chk@plt+0x4e10>
  407673:	48 8b 4e 40          	mov    0x40(%rsi),%rcx
  407677:	48 39 4f 40          	cmp    %rcx,0x40(%rdi)
  40767b:	48 8b 07             	mov    (%rdi),%rax
  40767e:	48 8b 16             	mov    (%rsi),%rdx
  407681:	7f 15                	jg     407698 <__sprintf_chk@plt+0x4e08>
  407683:	7c 1f                	jl     4076a4 <__sprintf_chk@plt+0x4e14>
  407685:	48 89 d6             	mov    %rdx,%rsi
  407688:	48 89 c7             	mov    %rax,%rdi
  40768b:	e9 c0 ae ff ff       	jmpq   402550 <strcmp@plt>
  407690:	84 c0                	test   %al,%al
  407692:	75 db                	jne    40766f <__sprintf_chk@plt+0x4ddf>
  407694:	0f 1f 40 00          	nopl   0x0(%rax)
  407698:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40769d:	c3                   	retq   
  40769e:	66 90                	xchg   %ax,%ax
  4076a0:	84 c0                	test   %al,%al
  4076a2:	74 cf                	je     407673 <__sprintf_chk@plt+0x4de3>
  4076a4:	b8 01 00 00 00       	mov    $0x1,%eax
  4076a9:	c3                   	retq   
  4076aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4076b0:	48 8b 4e 78          	mov    0x78(%rsi),%rcx
  4076b4:	48 39 4f 78          	cmp    %rcx,0x78(%rdi)
  4076b8:	48 8b 97 80 00 00 00 	mov    0x80(%rdi),%rdx
  4076bf:	48 8b 86 80 00 00 00 	mov    0x80(%rsi),%rax
  4076c6:	7f 18                	jg     4076e0 <__sprintf_chk@plt+0x4e50>
  4076c8:	7c 26                	jl     4076f0 <__sprintf_chk@plt+0x4e60>
  4076ca:	29 d0                	sub    %edx,%eax
  4076cc:	75 27                	jne    4076f5 <__sprintf_chk@plt+0x4e65>
  4076ce:	48 8b 36             	mov    (%rsi),%rsi
  4076d1:	48 8b 3f             	mov    (%rdi),%rdi
  4076d4:	e9 47 d9 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  4076d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4076e0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4076e5:	c3                   	retq   
  4076e6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4076ed:	00 00 00 
  4076f0:	b8 01 00 00 00       	mov    $0x1,%eax
  4076f5:	c3                   	retq   
  4076f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4076fd:	00 00 00 
  407700:	48 89 f0             	mov    %rsi,%rax
  407703:	48 8b 96 80 00 00 00 	mov    0x80(%rsi),%rdx
  40770a:	48 8b 77 78          	mov    0x78(%rdi),%rsi
  40770e:	48 39 70 78          	cmp    %rsi,0x78(%rax)
  407712:	48 8b 8f 80 00 00 00 	mov    0x80(%rdi),%rcx
  407719:	7f 15                	jg     407730 <__sprintf_chk@plt+0x4ea0>
  40771b:	7c 23                	jl     407740 <__sprintf_chk@plt+0x4eb0>
  40771d:	29 d1                	sub    %edx,%ecx
  40771f:	75 25                	jne    407746 <__sprintf_chk@plt+0x4eb6>
  407721:	48 8b 37             	mov    (%rdi),%rsi
  407724:	48 8b 38             	mov    (%rax),%rdi
  407727:	e9 f4 d8 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  40772c:	0f 1f 40 00          	nopl   0x0(%rax)
  407730:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  407735:	c3                   	retq   
  407736:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40773d:	00 00 00 
  407740:	b8 01 00 00 00       	mov    $0x1,%eax
  407745:	c3                   	retq   
  407746:	89 c8                	mov    %ecx,%eax
  407748:	c3                   	retq   
  407749:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407750:	48 8b 4e 58          	mov    0x58(%rsi),%rcx
  407754:	48 39 4f 58          	cmp    %rcx,0x58(%rdi)
  407758:	48 8b 57 60          	mov    0x60(%rdi),%rdx
  40775c:	48 8b 46 60          	mov    0x60(%rsi),%rax
  407760:	7f 16                	jg     407778 <__sprintf_chk@plt+0x4ee8>
  407762:	7c 1c                	jl     407780 <__sprintf_chk@plt+0x4ef0>
  407764:	29 d0                	sub    %edx,%eax
  407766:	75 1d                	jne    407785 <__sprintf_chk@plt+0x4ef5>
  407768:	48 8b 36             	mov    (%rsi),%rsi
  40776b:	48 8b 3f             	mov    (%rdi),%rdi
  40776e:	e9 ad d8 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  407773:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407778:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40777d:	c3                   	retq   
  40777e:	66 90                	xchg   %ax,%ax
  407780:	b8 01 00 00 00       	mov    $0x1,%eax
  407785:	c3                   	retq   
  407786:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40778d:	00 00 00 
  407790:	48 89 f0             	mov    %rsi,%rax
  407793:	48 8b 56 60          	mov    0x60(%rsi),%rdx
  407797:	48 8b 77 58          	mov    0x58(%rdi),%rsi
  40779b:	48 39 70 58          	cmp    %rsi,0x58(%rax)
  40779f:	48 8b 4f 60          	mov    0x60(%rdi),%rcx
  4077a3:	7f 1b                	jg     4077c0 <__sprintf_chk@plt+0x4f30>
  4077a5:	7c 29                	jl     4077d0 <__sprintf_chk@plt+0x4f40>
  4077a7:	29 d1                	sub    %edx,%ecx
  4077a9:	75 2b                	jne    4077d6 <__sprintf_chk@plt+0x4f46>
  4077ab:	48 8b 37             	mov    (%rdi),%rsi
  4077ae:	48 8b 38             	mov    (%rax),%rdi
  4077b1:	e9 6a d8 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  4077b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4077bd:	00 00 00 
  4077c0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4077c5:	c3                   	retq   
  4077c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4077cd:	00 00 00 
  4077d0:	b8 01 00 00 00       	mov    $0x1,%eax
  4077d5:	c3                   	retq   
  4077d6:	89 c8                	mov    %ecx,%eax
  4077d8:	c3                   	retq   
  4077d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4077e0:	48 8b 4e 68          	mov    0x68(%rsi),%rcx
  4077e4:	48 39 4f 68          	cmp    %rcx,0x68(%rdi)
  4077e8:	48 8b 57 70          	mov    0x70(%rdi),%rdx
  4077ec:	48 8b 46 70          	mov    0x70(%rsi),%rax
  4077f0:	7f 16                	jg     407808 <__sprintf_chk@plt+0x4f78>
  4077f2:	7c 1c                	jl     407810 <__sprintf_chk@plt+0x4f80>
  4077f4:	29 d0                	sub    %edx,%eax
  4077f6:	75 1d                	jne    407815 <__sprintf_chk@plt+0x4f85>
  4077f8:	48 8b 36             	mov    (%rsi),%rsi
  4077fb:	48 8b 3f             	mov    (%rdi),%rdi
  4077fe:	e9 1d d8 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  407803:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407808:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40780d:	c3                   	retq   
  40780e:	66 90                	xchg   %ax,%ax
  407810:	b8 01 00 00 00       	mov    $0x1,%eax
  407815:	c3                   	retq   
  407816:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40781d:	00 00 00 
  407820:	48 89 f0             	mov    %rsi,%rax
  407823:	48 8b 56 70          	mov    0x70(%rsi),%rdx
  407827:	48 8b 77 68          	mov    0x68(%rdi),%rsi
  40782b:	48 39 70 68          	cmp    %rsi,0x68(%rax)
  40782f:	48 8b 4f 70          	mov    0x70(%rdi),%rcx
  407833:	7f 1b                	jg     407850 <__sprintf_chk@plt+0x4fc0>
  407835:	7c 29                	jl     407860 <__sprintf_chk@plt+0x4fd0>
  407837:	29 d1                	sub    %edx,%ecx
  407839:	75 2b                	jne    407866 <__sprintf_chk@plt+0x4fd6>
  40783b:	48 8b 37             	mov    (%rdi),%rsi
  40783e:	48 8b 38             	mov    (%rax),%rdi
  407841:	e9 da d7 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  407846:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40784d:	00 00 00 
  407850:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  407855:	c3                   	retq   
  407856:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40785d:	00 00 00 
  407860:	b8 01 00 00 00       	mov    $0x1,%eax
  407865:	c3                   	retq   
  407866:	89 c8                	mov    %ecx,%eax
  407868:	c3                   	retq   
  407869:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407870:	55                   	push   %rbp
  407871:	48 89 f5             	mov    %rsi,%rbp
  407874:	53                   	push   %rbx
  407875:	48 89 fb             	mov    %rdi,%rbx
  407878:	48 81 ec a8 02 00 00 	sub    $0x2a8,%rsp
  40787f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  407886:	00 00 
  407888:	48 89 84 24 98 02 00 	mov    %rax,0x298(%rsp)
  40788f:	00 
  407890:	31 c0                	xor    %eax,%eax
  407892:	e8 99 f1 ff ff       	callq  406a30 <__sprintf_chk@plt+0x41a0>
  407897:	80 3d 76 38 21 00 00 	cmpb   $0x0,0x213876(%rip)        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  40789e:	0f 85 cc 00 00 00    	jne    407970 <__sprintf_chk@plt+0x50e0>
  4078a4:	80 3d 99 38 21 00 00 	cmpb   $0x0,0x213899(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  4078ab:	74 33                	je     4078e0 <__sprintf_chk@plt+0x5050>
  4078ad:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  4078b4:	b9 64 37 41 00       	mov    $0x413764,%ecx
  4078b9:	0f 85 01 01 00 00    	jne    4079c0 <__sprintf_chk@plt+0x5130>
  4078bf:	31 d2                	xor    %edx,%edx
  4078c1:	83 3d 88 38 21 00 04 	cmpl   $0x4,0x213888(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4078c8:	be 79 37 41 00       	mov    $0x413779,%esi
  4078cd:	0f 45 15 a0 38 21 00 	cmovne 0x2138a0(%rip),%edx        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  4078d4:	bf 01 00 00 00       	mov    $0x1,%edi
  4078d9:	31 c0                	xor    %eax,%eax
  4078db:	e8 50 ae ff ff       	callq  402730 <__printf_chk@plt>
  4078e0:	80 3d 96 38 21 00 00 	cmpb   $0x0,0x213896(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  4078e7:	74 28                	je     407911 <__sprintf_chk@plt+0x5081>
  4078e9:	31 d2                	xor    %edx,%edx
  4078eb:	83 3d 5e 38 21 00 04 	cmpl   $0x4,0x21385e(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4078f2:	48 8b 8b a8 00 00 00 	mov    0xa8(%rbx),%rcx
  4078f9:	0f 45 15 6c 38 21 00 	cmovne 0x21386c(%rip),%edx        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  407900:	be 79 37 41 00       	mov    $0x413779,%esi
  407905:	bf 01 00 00 00       	mov    $0x1,%edi
  40790a:	31 c0                	xor    %eax,%eax
  40790c:	e8 1f ae ff ff       	callq  402730 <__printf_chk@plt>
  407911:	48 89 e9             	mov    %rbp,%rcx
  407914:	31 d2                	xor    %edx,%edx
  407916:	31 f6                	xor    %esi,%esi
  407918:	48 89 df             	mov    %rbx,%rdi
  40791b:	e8 20 ec ff ff       	callq  406540 <__sprintf_chk@plt+0x3cb0>
  407920:	48 89 c5             	mov    %rax,%rbp
  407923:	8b 05 03 38 21 00    	mov    0x213803(%rip),%eax        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  407929:	85 c0                	test   %eax,%eax
  40792b:	74 1b                	je     407948 <__sprintf_chk@plt+0x50b8>
  40792d:	0f b6 bb b0 00 00 00 	movzbl 0xb0(%rbx),%edi
  407934:	8b 93 a0 00 00 00    	mov    0xa0(%rbx),%edx
  40793a:	8b 73 28             	mov    0x28(%rbx),%esi
  40793d:	e8 be e3 ff ff       	callq  405d00 <__sprintf_chk@plt+0x3470>
  407942:	0f b6 c0             	movzbl %al,%eax
  407945:	48 01 c5             	add    %rax,%rbp
  407948:	48 8b b4 24 98 02 00 	mov    0x298(%rsp),%rsi
  40794f:	00 
  407950:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  407957:	00 00 
  407959:	48 89 e8             	mov    %rbp,%rax
  40795c:	0f 85 84 00 00 00    	jne    4079e6 <__sprintf_chk@plt+0x5156>
  407962:	48 81 c4 a8 02 00 00 	add    $0x2a8,%rsp
  407969:	5b                   	pop    %rbx
  40796a:	5d                   	pop    %rbp
  40796b:	c3                   	retq   
  40796c:	0f 1f 40 00          	nopl   0x0(%rax)
  407970:	80 bb b0 00 00 00 00 	cmpb   $0x0,0xb0(%rbx)
  407977:	b9 64 37 41 00       	mov    $0x413764,%ecx
  40797c:	74 14                	je     407992 <__sprintf_chk@plt+0x5102>
  40797e:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
  407982:	48 85 ff             	test   %rdi,%rdi
  407985:	74 0b                	je     407992 <__sprintf_chk@plt+0x5102>
  407987:	48 89 e6             	mov    %rsp,%rsi
  40798a:	e8 e1 53 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  40798f:	48 89 c1             	mov    %rax,%rcx
  407992:	31 d2                	xor    %edx,%edx
  407994:	83 3d b5 37 21 00 04 	cmpl   $0x4,0x2137b5(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  40799b:	be 79 37 41 00       	mov    $0x413779,%esi
  4079a0:	0f 45 15 d1 37 21 00 	cmovne 0x2137d1(%rip),%edx        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  4079a7:	bf 01 00 00 00       	mov    $0x1,%edi
  4079ac:	31 c0                	xor    %eax,%eax
  4079ae:	e8 7d ad ff ff       	callq  402730 <__printf_chk@plt>
  4079b3:	e9 ec fe ff ff       	jmpq   4078a4 <__sprintf_chk@plt+0x5014>
  4079b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4079bf:	00 
  4079c0:	48 8b 7b 50          	mov    0x50(%rbx),%rdi
  4079c4:	4c 8b 05 6d 37 21 00 	mov    0x21376d(%rip),%r8        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  4079cb:	b9 00 02 00 00       	mov    $0x200,%ecx
  4079d0:	8b 15 6a 37 21 00    	mov    0x21376a(%rip),%edx        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  4079d6:	48 89 e6             	mov    %rsp,%rsi
  4079d9:	e8 92 43 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  4079de:	48 89 c1             	mov    %rax,%rcx
  4079e1:	e9 d9 fe ff ff       	jmpq   4078bf <__sprintf_chk@plt+0x502f>
  4079e6:	e8 b5 a9 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  4079eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4079f0:	83 3d 59 37 21 00 04 	cmpl   $0x4,0x213759(%rip)        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4079f7:	77 4d                	ja     407a46 <__sprintf_chk@plt+0x51b6>
  4079f9:	41 57                	push   %r15
  4079fb:	41 56                	push   %r14
  4079fd:	41 55                	push   %r13
  4079ff:	41 54                	push   %r12
  407a01:	55                   	push   %rbp
  407a02:	53                   	push   %rbx
  407a03:	48 83 ec 38          	sub    $0x38,%rsp
  407a07:	8b 05 43 37 21 00    	mov    0x213743(%rip),%eax        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  407a0d:	ff 24 c5 08 23 41 00 	jmpq   *0x412308(,%rax,8)
  407a14:	0f 1f 40 00          	nopl   0x0(%rax)
  407a18:	48 8b 3d f1 2b 21 00 	mov    0x212bf1(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407a1f:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407a23:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407a27:	0f 83 0c 04 00 00    	jae    407e39 <__sprintf_chk@plt+0x55a9>
  407a2d:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407a31:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407a35:	c6 00 0a             	movb   $0xa,(%rax)
  407a38:	48 83 c4 38          	add    $0x38,%rsp
  407a3c:	5b                   	pop    %rbx
  407a3d:	5d                   	pop    %rbp
  407a3e:	41 5c                	pop    %r12
  407a40:	41 5d                	pop    %r13
  407a42:	41 5e                	pop    %r14
  407a44:	41 5f                	pop    %r15
  407a46:	f3 c3                	repz retq 
  407a48:	31 ff                	xor    %edi,%edi
  407a4a:	45 31 ff             	xor    %r15d,%r15d
  407a4d:	e8 7e e4 ff ff       	callq  405ed0 <__sprintf_chk@plt+0x3640>
  407a52:	48 8d 14 40          	lea    (%rax,%rax,2),%rdx
  407a56:	49 89 c5             	mov    %rax,%r13
  407a59:	48 8b 05 c8 35 21 00 	mov    0x2135c8(%rip),%rax        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  407a60:	4c 8d 74 d0 e8       	lea    -0x18(%rax,%rdx,8),%r14
  407a65:	48 8b 05 3c 37 21 00 	mov    0x21373c(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407a6c:	48 8b 18             	mov    (%rax),%rbx
  407a6f:	48 89 df             	mov    %rbx,%rdi
  407a72:	e8 d9 e2 ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  407a77:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  407a7c:	49 8b 46 10          	mov    0x10(%r14),%rax
  407a80:	31 f6                	xor    %esi,%esi
  407a82:	48 89 df             	mov    %rbx,%rdi
  407a85:	bb 01 00 00 00       	mov    $0x1,%ebx
  407a8a:	4c 8b 20             	mov    (%rax),%r12
  407a8d:	e8 de fd ff ff       	callq  407870 <__sprintf_chk@plt+0x4fe0>
  407a92:	48 83 3d 16 37 21 00 	cmpq   $0x1,0x213716(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407a99:	01 
  407a9a:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  407a9f:	77 6c                	ja     407b0d <__sprintf_chk@plt+0x527d>
  407aa1:	e9 72 ff ff ff       	jmpq   407a18 <__sprintf_chk@plt+0x5188>
  407aa6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  407aad:	00 00 00 
  407ab0:	48 8b 3d 59 2b 21 00 	mov    0x212b59(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407ab7:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407abb:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407abf:	0f 83 62 03 00 00    	jae    407e27 <__sprintf_chk@plt+0x5597>
  407ac5:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407ac9:	45 31 ff             	xor    %r15d,%r15d
  407acc:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407ad0:	c6 00 0a             	movb   $0xa,(%rax)
  407ad3:	48 8b 05 ce 36 21 00 	mov    0x2136ce(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407ada:	4c 89 fe             	mov    %r15,%rsi
  407add:	4c 8b 24 d8          	mov    (%rax,%rbx,8),%r12
  407ae1:	48 83 c3 01          	add    $0x1,%rbx
  407ae5:	4c 89 e7             	mov    %r12,%rdi
  407ae8:	e8 83 fd ff ff       	callq  407870 <__sprintf_chk@plt+0x4fe0>
  407aed:	4c 89 e7             	mov    %r12,%rdi
  407af0:	e8 5b e2 ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  407af5:	48 3b 1d b4 36 21 00 	cmp    0x2136b4(%rip),%rbx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407afc:	49 8b 56 10          	mov    0x10(%r14),%rdx
  407b00:	48 89 c1             	mov    %rax,%rcx
  407b03:	4c 8b 24 ea          	mov    (%rdx,%rbp,8),%r12
  407b07:	0f 83 0b ff ff ff    	jae    407a18 <__sprintf_chk@plt+0x5188>
  407b0d:	31 d2                	xor    %edx,%edx
  407b0f:	48 89 d8             	mov    %rbx,%rax
  407b12:	49 f7 f5             	div    %r13
  407b15:	48 85 d2             	test   %rdx,%rdx
  407b18:	48 89 d5             	mov    %rdx,%rbp
  407b1b:	74 93                	je     407ab0 <__sprintf_chk@plt+0x5220>
  407b1d:	4d 01 fc             	add    %r15,%r12
  407b20:	4a 8d 3c 39          	lea    (%rcx,%r15,1),%rdi
  407b24:	4c 89 e6             	mov    %r12,%rsi
  407b27:	4d 89 e7             	mov    %r12,%r15
  407b2a:	e8 d1 d6 ff ff       	callq  405200 <__sprintf_chk@plt+0x2970>
  407b2f:	eb a2                	jmp    407ad3 <__sprintf_chk@plt+0x5243>
  407b31:	48 83 3d 77 36 21 00 	cmpq   $0x0,0x213677(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407b38:	00 
  407b39:	0f 84 d9 fe ff ff    	je     407a18 <__sprintf_chk@plt+0x5188>
  407b3f:	48 8b 05 62 36 21 00 	mov    0x213662(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407b46:	31 db                	xor    %ebx,%ebx
  407b48:	45 31 e4             	xor    %r12d,%r12d
  407b4b:	4c 8b 28             	mov    (%rax),%r13
  407b4e:	4c 89 ef             	mov    %r13,%rdi
  407b51:	e8 fa e1 ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  407b56:	48 89 c5             	mov    %rax,%rbp
  407b59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407b60:	4c 89 e6             	mov    %r12,%rsi
  407b63:	4c 89 ef             	mov    %r13,%rdi
  407b66:	48 83 c3 01          	add    $0x1,%rbx
  407b6a:	e8 01 fd ff ff       	callq  407870 <__sprintf_chk@plt+0x4fe0>
  407b6f:	48 3b 1d 3a 36 21 00 	cmp    0x21363a(%rip),%rbx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407b76:	49 89 ec             	mov    %rbp,%r12
  407b79:	0f 83 99 fe ff ff    	jae    407a18 <__sprintf_chk@plt+0x5188>
  407b7f:	48 8b 05 22 36 21 00 	mov    0x213622(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407b86:	4c 8b 2c d8          	mov    (%rax,%rbx,8),%r13
  407b8a:	4c 89 ef             	mov    %r13,%rdi
  407b8d:	e8 be e1 ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  407b92:	48 85 db             	test   %rbx,%rbx
  407b95:	0f 84 65 02 00 00    	je     407e00 <__sprintf_chk@plt+0x5570>
  407b9b:	4c 8d 65 02          	lea    0x2(%rbp),%r12
  407b9f:	4a 8d 2c 20          	lea    (%rax,%r12,1),%rbp
  407ba3:	48 3b 2d 1e 35 21 00 	cmp    0x21351e(%rip),%rbp        # 61b0c8 <stderr@@GLIBC_2.2.5+0xa78>
  407baa:	0f 83 60 02 00 00    	jae    407e10 <__sprintf_chk@plt+0x5580>
  407bb0:	41 bf 20 00 00 00    	mov    $0x20,%r15d
  407bb6:	41 be 20 00 00 00    	mov    $0x20,%r14d
  407bbc:	48 8b 3d 4d 2a 21 00 	mov    0x212a4d(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407bc3:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407bc7:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407bcb:	0f 83 ba 02 00 00    	jae    407e8b <__sprintf_chk@plt+0x55fb>
  407bd1:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407bd5:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407bd9:	c6 00 2c             	movb   $0x2c,(%rax)
  407bdc:	48 8b 3d 2d 2a 21 00 	mov    0x212a2d(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407be3:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407be7:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407beb:	0f 83 8d 02 00 00    	jae    407e7e <__sprintf_chk@plt+0x55ee>
  407bf1:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407bf5:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407bf9:	44 88 30             	mov    %r14b,(%rax)
  407bfc:	e9 5f ff ff ff       	jmpq   407b60 <__sprintf_chk@plt+0x52d0>
  407c01:	31 db                	xor    %ebx,%ebx
  407c03:	48 83 3d a5 35 21 00 	cmpq   $0x0,0x2135a5(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407c0a:	00 
  407c0b:	0f 84 27 fe ff ff    	je     407a38 <__sprintf_chk@plt+0x51a8>
  407c11:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407c18:	e8 13 ee ff ff       	callq  406a30 <__sprintf_chk@plt+0x41a0>
  407c1d:	48 8b 05 84 35 21 00 	mov    0x213584(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407c24:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
  407c28:	e8 43 ef ff ff       	callq  406b70 <__sprintf_chk@plt+0x42e0>
  407c2d:	48 8b 3d dc 29 21 00 	mov    0x2129dc(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407c34:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407c38:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407c3c:	0f 83 2d 02 00 00    	jae    407e6f <__sprintf_chk@plt+0x55df>
  407c42:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407c46:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407c4a:	c6 00 0a             	movb   $0xa,(%rax)
  407c4d:	48 83 05 c3 33 21 00 	addq   $0x1,0x2133c3(%rip)        # 61b018 <stderr@@GLIBC_2.2.5+0x9c8>
  407c54:	01 
  407c55:	48 83 c3 01          	add    $0x1,%rbx
  407c59:	48 39 1d 50 35 21 00 	cmp    %rbx,0x213550(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407c60:	77 b6                	ja     407c18 <__sprintf_chk@plt+0x5388>
  407c62:	48 83 c4 38          	add    $0x38,%rsp
  407c66:	5b                   	pop    %rbx
  407c67:	5d                   	pop    %rbp
  407c68:	41 5c                	pop    %r12
  407c6a:	41 5d                	pop    %r13
  407c6c:	41 5e                	pop    %r14
  407c6e:	41 5f                	pop    %r15
  407c70:	e9 d1 fd ff ff       	jmpq   407a46 <__sprintf_chk@plt+0x51b6>
  407c75:	31 db                	xor    %ebx,%ebx
  407c77:	48 83 3d 31 35 21 00 	cmpq   $0x0,0x213531(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407c7e:	00 
  407c7f:	0f 84 b3 fd ff ff    	je     407a38 <__sprintf_chk@plt+0x51a8>
  407c85:	0f 1f 00             	nopl   (%rax)
  407c88:	48 8b 05 19 35 21 00 	mov    0x213519(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407c8f:	31 f6                	xor    %esi,%esi
  407c91:	48 8b 3c d8          	mov    (%rax,%rbx,8),%rdi
  407c95:	e8 d6 fb ff ff       	callq  407870 <__sprintf_chk@plt+0x4fe0>
  407c9a:	48 8b 3d 6f 29 21 00 	mov    0x21296f(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407ca1:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407ca5:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407ca9:	0f 83 b1 01 00 00    	jae    407e60 <__sprintf_chk@plt+0x55d0>
  407caf:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407cb3:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407cb7:	c6 00 0a             	movb   $0xa,(%rax)
  407cba:	48 83 c3 01          	add    $0x1,%rbx
  407cbe:	48 39 1d eb 34 21 00 	cmp    %rbx,0x2134eb(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407cc5:	77 c1                	ja     407c88 <__sprintf_chk@plt+0x53f8>
  407cc7:	48 83 c4 38          	add    $0x38,%rsp
  407ccb:	5b                   	pop    %rbx
  407ccc:	5d                   	pop    %rbp
  407ccd:	41 5c                	pop    %r12
  407ccf:	41 5d                	pop    %r13
  407cd1:	41 5e                	pop    %r14
  407cd3:	41 5f                	pop    %r15
  407cd5:	e9 6c fd ff ff       	jmpq   407a46 <__sprintf_chk@plt+0x51b6>
  407cda:	bf 01 00 00 00       	mov    $0x1,%edi
  407cdf:	e8 ec e1 ff ff       	callq  405ed0 <__sprintf_chk@plt+0x3640>
  407ce4:	48 8d 14 40          	lea    (%rax,%rax,2),%rdx
  407ce8:	48 89 c1             	mov    %rax,%rcx
  407ceb:	48 8b 05 36 33 21 00 	mov    0x213336(%rip),%rax        # 61b028 <stderr@@GLIBC_2.2.5+0x9d8>
  407cf2:	4c 8d 7c d0 e8       	lea    -0x18(%rax,%rdx,8),%r15
  407cf7:	48 8b 05 b2 34 21 00 	mov    0x2134b2(%rip),%rax        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407cfe:	31 d2                	xor    %edx,%edx
  407d00:	48 f7 f1             	div    %rcx
  407d03:	48 85 d2             	test   %rdx,%rdx
  407d06:	0f 95 c2             	setne  %dl
  407d09:	0f b6 d2             	movzbl %dl,%edx
  407d0c:	48 01 c2             	add    %rax,%rdx
  407d0f:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  407d14:	0f 84 1e fd ff ff    	je     407a38 <__sprintf_chk@plt+0x51a8>
  407d1a:	48 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%rax
  407d21:	00 
  407d22:	48 c7 44 24 28 00 00 	movq   $0x0,0x28(%rsp)
  407d29:	00 00 
  407d2b:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  407d30:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  407d35:	31 ed                	xor    %ebp,%ebp
  407d37:	31 db                	xor    %ebx,%ebx
  407d39:	4c 8d 2c c5 00 00 00 	lea    0x0(,%rax,8),%r13
  407d40:	00 
  407d41:	49 89 c4             	mov    %rax,%r12
  407d44:	eb 25                	jmp    407d6b <__sprintf_chk@plt+0x54db>
  407d46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  407d4d:	00 00 00 
  407d50:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
  407d55:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  407d5a:	49 01 de             	add    %rbx,%r14
  407d5d:	48 01 df             	add    %rbx,%rdi
  407d60:	4c 89 f6             	mov    %r14,%rsi
  407d63:	4c 89 f3             	mov    %r14,%rbx
  407d66:	e8 95 d4 ff ff       	callq  405200 <__sprintf_chk@plt+0x2970>
  407d6b:	48 8b 05 36 34 21 00 	mov    0x213436(%rip),%rax        # 61b1a8 <stderr@@GLIBC_2.2.5+0xb58>
  407d72:	4e 8b 34 28          	mov    (%rax,%r13,1),%r14
  407d76:	4c 89 f7             	mov    %r14,%rdi
  407d79:	e8 d2 df ff ff       	callq  405d50 <__sprintf_chk@plt+0x34c0>
  407d7e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  407d83:	49 8b 47 10          	mov    0x10(%r15),%rax
  407d87:	48 89 de             	mov    %rbx,%rsi
  407d8a:	4c 89 f7             	mov    %r14,%rdi
  407d8d:	48 8b 0c 28          	mov    (%rax,%rbp,1),%rcx
  407d91:	48 83 c5 08          	add    $0x8,%rbp
  407d95:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
  407d9a:	e8 d1 fa ff ff       	callq  407870 <__sprintf_chk@plt+0x4fe0>
  407d9f:	4c 03 64 24 18       	add    0x18(%rsp),%r12
  407da4:	4c 03 6c 24 20       	add    0x20(%rsp),%r13
  407da9:	4c 3b 25 00 34 21 00 	cmp    0x213400(%rip),%r12        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407db0:	72 9e                	jb     407d50 <__sprintf_chk@plt+0x54c0>
  407db2:	48 8b 3d 57 28 21 00 	mov    0x212857(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  407db9:	48 8b 47 28          	mov    0x28(%rdi),%rax
  407dbd:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  407dc1:	0f 83 8a 00 00 00    	jae    407e51 <__sprintf_chk@plt+0x55c1>
  407dc7:	48 8d 50 01          	lea    0x1(%rax),%rdx
  407dcb:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  407dcf:	c6 00 0a             	movb   $0xa,(%rax)
  407dd2:	48 83 44 24 28 01    	addq   $0x1,0x28(%rsp)
  407dd8:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  407ddd:	48 39 44 24 28       	cmp    %rax,0x28(%rsp)
  407de2:	0f 85 48 ff ff ff    	jne    407d30 <__sprintf_chk@plt+0x54a0>
  407de8:	48 83 c4 38          	add    $0x38,%rsp
  407dec:	5b                   	pop    %rbx
  407ded:	5d                   	pop    %rbp
  407dee:	41 5c                	pop    %r12
  407df0:	41 5d                	pop    %r13
  407df2:	41 5e                	pop    %r14
  407df4:	41 5f                	pop    %r15
  407df6:	e9 4b fc ff ff       	jmpq   407a46 <__sprintf_chk@plt+0x51b6>
  407dfb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  407e00:	48 01 c5             	add    %rax,%rbp
  407e03:	e9 58 fd ff ff       	jmpq   407b60 <__sprintf_chk@plt+0x52d0>
  407e08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  407e0f:	00 
  407e10:	41 bf 0a 00 00 00    	mov    $0xa,%r15d
  407e16:	41 be 0a 00 00 00    	mov    $0xa,%r14d
  407e1c:	45 31 e4             	xor    %r12d,%r12d
  407e1f:	48 89 c5             	mov    %rax,%rbp
  407e22:	e9 95 fd ff ff       	jmpq   407bbc <__sprintf_chk@plt+0x532c>
  407e27:	be 0a 00 00 00       	mov    $0xa,%esi
  407e2c:	45 31 ff             	xor    %r15d,%r15d
  407e2f:	e8 cc a5 ff ff       	callq  402400 <__overflow@plt>
  407e34:	e9 9a fc ff ff       	jmpq   407ad3 <__sprintf_chk@plt+0x5243>
  407e39:	48 83 c4 38          	add    $0x38,%rsp
  407e3d:	be 0a 00 00 00       	mov    $0xa,%esi
  407e42:	5b                   	pop    %rbx
  407e43:	5d                   	pop    %rbp
  407e44:	41 5c                	pop    %r12
  407e46:	41 5d                	pop    %r13
  407e48:	41 5e                	pop    %r14
  407e4a:	41 5f                	pop    %r15
  407e4c:	e9 af a5 ff ff       	jmpq   402400 <__overflow@plt>
  407e51:	be 0a 00 00 00       	mov    $0xa,%esi
  407e56:	e8 a5 a5 ff ff       	callq  402400 <__overflow@plt>
  407e5b:	e9 72 ff ff ff       	jmpq   407dd2 <__sprintf_chk@plt+0x5542>
  407e60:	be 0a 00 00 00       	mov    $0xa,%esi
  407e65:	e8 96 a5 ff ff       	callq  402400 <__overflow@plt>
  407e6a:	e9 4b fe ff ff       	jmpq   407cba <__sprintf_chk@plt+0x542a>
  407e6f:	be 0a 00 00 00       	mov    $0xa,%esi
  407e74:	e8 87 a5 ff ff       	callq  402400 <__overflow@plt>
  407e79:	e9 cf fd ff ff       	jmpq   407c4d <__sprintf_chk@plt+0x53bd>
  407e7e:	44 89 fe             	mov    %r15d,%esi
  407e81:	e8 7a a5 ff ff       	callq  402400 <__overflow@plt>
  407e86:	e9 d5 fc ff ff       	jmpq   407b60 <__sprintf_chk@plt+0x52d0>
  407e8b:	be 2c 00 00 00       	mov    $0x2c,%esi
  407e90:	e8 6b a5 ff ff       	callq  402400 <__overflow@plt>
  407e95:	e9 42 fd ff ff       	jmpq   407bdc <__sprintf_chk@plt+0x534c>
  407e9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  407ea0:	55                   	push   %rbp
  407ea1:	48 89 e5             	mov    %rsp,%rbp
  407ea4:	41 57                	push   %r15
  407ea6:	41 56                	push   %r14
  407ea8:	41 55                	push   %r13
  407eaa:	41 89 f5             	mov    %esi,%r13d
  407ead:	41 54                	push   %r12
  407eaf:	53                   	push   %rbx
  407eb0:	48 89 fb             	mov    %rdi,%rbx
  407eb3:	48 81 ec 78 03 00 00 	sub    $0x378,%rsp
  407eba:	89 95 7c fc ff ff    	mov    %edx,-0x384(%rbp)
  407ec0:	48 89 ca             	mov    %rcx,%rdx
  407ec3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  407eca:	00 00 
  407ecc:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
  407ed0:	31 c0                	xor    %eax,%eax
  407ed2:	48 8b 0d d7 32 21 00 	mov    0x2132d7(%rip),%rcx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  407ed9:	48 3b 0d d8 32 21 00 	cmp    0x2132d8(%rip),%rcx        # 61b1b8 <stderr@@GLIBC_2.2.5+0xb68>
  407ee0:	4c 8b 0d d9 32 21 00 	mov    0x2132d9(%rip),%r9        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  407ee7:	0f 84 13 05 00 00    	je     408400 <__sprintf_chk@plt+0x5b70>
  407eed:	48 8d 0c 49          	lea    (%rcx,%rcx,2),%rcx
  407ef1:	be c0 00 00 00       	mov    $0xc0,%esi
  407ef6:	48 c1 e1 06          	shl    $0x6,%rcx
  407efa:	4d 8d 34 09          	lea    (%r9,%rcx,1),%r14
  407efe:	41 f6 c6 01          	test   $0x1,%r14b
  407f02:	4c 89 f7             	mov    %r14,%rdi
  407f05:	0f 85 0d 0b 00 00    	jne    408a18 <__sprintf_chk@plt+0x6188>
  407f0b:	40 f6 c7 02          	test   $0x2,%dil
  407f0f:	0f 85 cb 0a 00 00    	jne    4089e0 <__sprintf_chk@plt+0x6150>
  407f15:	40 f6 c7 04          	test   $0x4,%dil
  407f19:	0f 85 e1 0a 00 00    	jne    408a00 <__sprintf_chk@plt+0x6170>
  407f1f:	89 f1                	mov    %esi,%ecx
  407f21:	31 c0                	xor    %eax,%eax
  407f23:	c1 e9 03             	shr    $0x3,%ecx
  407f26:	40 f6 c6 04          	test   $0x4,%sil
  407f2a:	f3 48 ab             	rep stos %rax,%es:(%rdi)
  407f2d:	0f 85 2d 04 00 00    	jne    408360 <__sprintf_chk@plt+0x5ad0>
  407f33:	40 f6 c6 02          	test   $0x2,%sil
  407f37:	0f 85 03 04 00 00    	jne    408340 <__sprintf_chk@plt+0x5ab0>
  407f3d:	83 e6 01             	and    $0x1,%esi
  407f40:	0f 85 f2 03 00 00    	jne    408338 <__sprintf_chk@plt+0x5aa8>
  407f46:	80 bd 7c fc ff ff 00 	cmpb   $0x0,-0x384(%rbp)
  407f4d:	49 c7 46 18 00 00 00 	movq   $0x0,0x18(%r14)
  407f54:	00 
  407f55:	45 89 ae a0 00 00 00 	mov    %r13d,0xa0(%r14)
  407f5c:	0f 84 be 02 00 00    	je     408220 <__sprintf_chk@plt+0x5990>
  407f62:	8b 0d a8 31 21 00    	mov    0x2131a8(%rip),%ecx        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  407f68:	0f b6 33             	movzbl (%rbx),%esi
  407f6b:	49 89 dc             	mov    %rbx,%r12
  407f6e:	40 80 fe 2f          	cmp    $0x2f,%sil
  407f72:	74 0d                	je     407f81 <__sprintf_chk@plt+0x56f1>
  407f74:	44 0f b6 3a          	movzbl (%rdx),%r15d
  407f78:	45 84 ff             	test   %r15b,%r15b
  407f7b:	0f 85 ef 06 00 00    	jne    408670 <__sprintf_chk@plt+0x5de0>
  407f81:	83 f9 03             	cmp    $0x3,%ecx
  407f84:	72 77                	jb     407ffd <__sprintf_chk@plt+0x576d>
  407f86:	83 f9 04             	cmp    $0x4,%ecx
  407f89:	76 65                	jbe    407ff0 <__sprintf_chk@plt+0x5760>
  407f8b:	83 f9 05             	cmp    $0x5,%ecx
  407f8e:	66 90                	xchg   %ax,%ax
  407f90:	75 6b                	jne    407ffd <__sprintf_chk@plt+0x576d>
  407f92:	4d 8d 7e 10          	lea    0x10(%r14),%r15
  407f96:	4c 89 e6             	mov    %r12,%rsi
  407f99:	bf 01 00 00 00       	mov    $0x1,%edi
  407f9e:	4c 89 fa             	mov    %r15,%rdx
  407fa1:	e8 6a a6 ff ff       	callq  402610 <__xstat@plt>
  407fa6:	89 c2                	mov    %eax,%edx
  407fa8:	be 01 00 00 00       	mov    $0x1,%esi
  407fad:	85 d2                	test   %edx,%edx
  407faf:	74 68                	je     408019 <__sprintf_chk@plt+0x5789>
  407fb1:	31 ff                	xor    %edi,%edi
  407fb3:	ba 05 00 00 00       	mov    $0x5,%edx
  407fb8:	be 83 37 41 00       	mov    $0x413783,%esi
  407fbd:	e8 9e a3 ff ff       	callq  402360 <dcgettext@plt>
  407fc2:	44 8b bd 7c fc ff ff 	mov    -0x384(%rbp),%r15d
  407fc9:	4c 89 e2             	mov    %r12,%rdx
  407fcc:	48 89 c6             	mov    %rax,%rsi
  407fcf:	45 31 e4             	xor    %r12d,%r12d
  407fd2:	41 0f b6 ff          	movzbl %r15b,%edi
  407fd6:	e8 35 d8 ff ff       	callq  405810 <__sprintf_chk@plt+0x2f80>
  407fdb:	45 84 ff             	test   %r15b,%r15b
  407fde:	0f 85 10 02 00 00    	jne    4081f4 <__sprintf_chk@plt+0x5964>
  407fe4:	e9 f8 01 00 00       	jmpq   4081e1 <__sprintf_chk@plt+0x5951>
  407fe9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  407ff0:	80 bd 7c fc ff ff 00 	cmpb   $0x0,-0x384(%rbp)
  407ff7:	0f 85 b3 08 00 00    	jne    4088b0 <__sprintf_chk@plt+0x6020>
  407ffd:	4d 8d 7e 10          	lea    0x10(%r14),%r15
  408001:	4c 89 fa             	mov    %r15,%rdx
  408004:	4c 89 e6             	mov    %r12,%rsi
  408007:	bf 01 00 00 00       	mov    $0x1,%edi
  40800c:	e8 7f a3 ff ff       	callq  402390 <__lxstat@plt>
  408011:	31 f6                	xor    %esi,%esi
  408013:	89 c2                	mov    %eax,%edx
  408015:	85 d2                	test   %edx,%edx
  408017:	75 98                	jne    407fb1 <__sprintf_chk@plt+0x5721>
  408019:	41 83 fd 05          	cmp    $0x5,%r13d
  40801d:	41 c6 86 b0 00 00 00 	movb   $0x1,0xb0(%r14)
  408024:	01 
  408025:	0f 84 55 03 00 00    	je     408380 <__sprintf_chk@plt+0x5af0>
  40802b:	41 8b 46 28          	mov    0x28(%r14),%eax
  40802f:	25 00 f0 00 00       	and    $0xf000,%eax
  408034:	3d 00 80 00 00       	cmp    $0x8000,%eax
  408039:	0f 84 41 03 00 00    	je     408380 <__sprintf_chk@plt+0x5af0>
  40803f:	8b 0d 0b 31 21 00    	mov    0x21310b(%rip),%ecx        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  408045:	85 c9                	test   %ecx,%ecx
  408047:	74 0d                	je     408056 <__sprintf_chk@plt+0x57c6>
  408049:	80 3d 2d 31 21 00 00 	cmpb   $0x0,0x21312d(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  408050:	0f 84 99 00 00 00    	je     4080ef <__sprintf_chk@plt+0x585f>
  408056:	48 8b 05 13 26 21 00 	mov    0x212613(%rip),%rax        # 61a670 <stderr@@GLIBC_2.2.5+0x20>
  40805d:	49 39 46 10          	cmp    %rax,0x10(%r14)
  408061:	89 95 70 fc ff ff    	mov    %edx,-0x390(%rbp)
  408067:	0f 84 e5 0b 00 00    	je     408c52 <__sprintf_chk@plt+0x63c2>
  40806d:	40 84 f6             	test   %sil,%sil
  408070:	4c 89 e7             	mov    %r12,%rdi
  408073:	49 8d b6 a8 00 00 00 	lea    0xa8(%r14),%rsi
  40807a:	0f 84 d8 03 00 00    	je     408458 <__sprintf_chk@plt+0x5bc8>
  408080:	e8 9b 97 00 00       	callq  411820 <__sprintf_chk@plt+0xef90>
  408085:	85 c0                	test   %eax,%eax
  408087:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  40808d:	0f 88 d8 03 00 00    	js     40846b <__sprintf_chk@plt+0x5bdb>
  408093:	49 8b be a8 00 00 00 	mov    0xa8(%r14),%rdi
  40809a:	be b1 37 41 00       	mov    $0x4137b1,%esi
  40809f:	b9 0a 00 00 00       	mov    $0xa,%ecx
  4080a4:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  4080a6:	41 0f 95 c5          	setne  %r13b
  4080aa:	8b 05 a0 30 21 00    	mov    0x2130a0(%rip),%eax        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4080b0:	85 c0                	test   %eax,%eax
  4080b2:	0f 84 78 07 00 00    	je     408830 <__sprintf_chk@plt+0x5fa0>
  4080b8:	31 c9                	xor    %ecx,%ecx
  4080ba:	89 ce                	mov    %ecx,%esi
  4080bc:	31 c0                	xor    %eax,%eax
  4080be:	44 08 ee             	or     %r13b,%sil
  4080c1:	74 16                	je     4080d9 <__sprintf_chk@plt+0x5849>
  4080c3:	83 f1 01             	xor    $0x1,%ecx
  4080c6:	41 20 cd             	and    %cl,%r13b
  4080c9:	44 89 e8             	mov    %r13d,%eax
  4080cc:	41 0f 45 f5          	cmovne %r13d,%esi
  4080d0:	c1 e0 1f             	shl    $0x1f,%eax
  4080d3:	c1 f8 1f             	sar    $0x1f,%eax
  4080d6:	83 c0 02             	add    $0x2,%eax
  4080d9:	40 08 35 9c 30 21 00 	or     %sil,0x21309c(%rip)        # 61b17c <stderr@@GLIBC_2.2.5+0xb2c>
  4080e0:	85 d2                	test   %edx,%edx
  4080e2:	41 89 86 b4 00 00 00 	mov    %eax,0xb4(%r14)
  4080e9:	0f 85 d0 03 00 00    	jne    4084bf <__sprintf_chk@plt+0x5c2f>
  4080ef:	41 8b 46 28          	mov    0x28(%r14),%eax
  4080f3:	25 00 f0 00 00       	and    $0xf000,%eax
  4080f8:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  4080fd:	0f 84 fd 03 00 00    	je     408500 <__sprintf_chk@plt+0x5c70>
  408103:	3d 00 40 00 00       	cmp    $0x4000,%eax
  408108:	0f 84 6a 08 00 00    	je     408978 <__sprintf_chk@plt+0x60e8>
  40810e:	44 8b 2d 3b 30 21 00 	mov    0x21303b(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  408115:	41 c7 86 a0 00 00 00 	movl   $0x5,0xa0(%r14)
  40811c:	05 00 00 00 
  408120:	45 85 ed             	test   %r13d,%r13d
  408123:	4d 8b 66 50          	mov    0x50(%r14),%r12
  408127:	74 09                	je     408132 <__sprintf_chk@plt+0x58a2>
  408129:	80 3d 14 30 21 00 00 	cmpb   $0x0,0x213014(%rip)        # 61b144 <stderr@@GLIBC_2.2.5+0xaf4>
  408130:	74 76                	je     4081a8 <__sprintf_chk@plt+0x5918>
  408132:	4c 8b 05 ff 2f 21 00 	mov    0x212fff(%rip),%r8        # 61b138 <stderr@@GLIBC_2.2.5+0xae8>
  408139:	8b 15 01 30 21 00    	mov    0x213001(%rip),%edx        # 61b140 <stderr@@GLIBC_2.2.5+0xaf0>
  40813f:	4c 8d bd 30 fd ff ff 	lea    -0x2d0(%rbp),%r15
  408146:	b9 00 02 00 00       	mov    $0x200,%ecx
  40814b:	4c 89 e7             	mov    %r12,%rdi
  40814e:	4c 89 fe             	mov    %r15,%rsi
  408151:	e8 1a 3c 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  408156:	31 f6                	xor    %esi,%esi
  408158:	48 89 c7             	mov    %rax,%rdi
  40815b:	e8 c0 52 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  408160:	3b 05 0e 30 21 00    	cmp    0x21300e(%rip),%eax        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  408166:	7e 06                	jle    40816e <__sprintf_chk@plt+0x58de>
  408168:	89 05 06 30 21 00    	mov    %eax,0x213006(%rip)        # 61b174 <stderr@@GLIBC_2.2.5+0xb24>
  40816e:	44 8b 2d db 2f 21 00 	mov    0x212fdb(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  408175:	45 85 ed             	test   %r13d,%r13d
  408178:	75 2e                	jne    4081a8 <__sprintf_chk@plt+0x5918>
  40817a:	80 3d e8 23 21 00 00 	cmpb   $0x0,0x2123e8(%rip)        # 61a569 <_fini@@Base+0x20866d>
  408181:	0f 85 89 09 00 00    	jne    408b10 <__sprintf_chk@plt+0x6280>
  408187:	80 3d da 23 21 00 00 	cmpb   $0x0,0x2123da(%rip)        # 61a568 <_fini@@Base+0x20866c>
  40818e:	0f 85 fc 08 00 00    	jne    408a90 <__sprintf_chk@plt+0x6200>
  408194:	80 3d ab 2f 21 00 00 	cmpb   $0x0,0x212fab(%rip)        # 61b146 <stderr@@GLIBC_2.2.5+0xaf6>
  40819b:	0f 85 cf 08 00 00    	jne    408a70 <__sprintf_chk@plt+0x61e0>
  4081a1:	44 8b 2d a8 2f 21 00 	mov    0x212fa8(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4081a8:	80 3d ce 2f 21 00 00 	cmpb   $0x0,0x212fce(%rip)        # 61b17d <stderr@@GLIBC_2.2.5+0xb2d>
  4081af:	74 18                	je     4081c9 <__sprintf_chk@plt+0x5939>
  4081b1:	49 8b be a8 00 00 00 	mov    0xa8(%r14),%rdi
  4081b8:	e8 c3 a1 ff ff       	callq  402380 <strlen@plt>
  4081bd:	3b 05 a9 2f 21 00    	cmp    0x212fa9(%rip),%eax        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  4081c3:	0f 8f 67 05 00 00    	jg     408730 <__sprintf_chk@plt+0x5ea0>
  4081c9:	45 85 ed             	test   %r13d,%r13d
  4081cc:	0f 84 6d 05 00 00    	je     40873f <__sprintf_chk@plt+0x5eaf>
  4081d2:	0f b6 05 3b 2f 21 00 	movzbl 0x212f3b(%rip),%eax        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  4081d9:	84 c0                	test   %al,%al
  4081db:	0f 85 5f 04 00 00    	jne    408640 <__sprintf_chk@plt+0x5db0>
  4081e1:	48 89 df             	mov    %rbx,%rdi
  4081e4:	e8 47 8c 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  4081e9:	48 83 05 bf 2f 21 00 	addq   $0x1,0x212fbf(%rip)        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  4081f0:	01 
  4081f1:	49 89 06             	mov    %rax,(%r14)
  4081f4:	48 8b 5d c8          	mov    -0x38(%rbp),%rbx
  4081f8:	64 48 33 1c 25 28 00 	xor    %fs:0x28,%rbx
  4081ff:	00 00 
  408201:	4c 89 e0             	mov    %r12,%rax
  408204:	0f 85 6c 0a 00 00    	jne    408c76 <__sprintf_chk@plt+0x63e6>
  40820a:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
  40820e:	5b                   	pop    %rbx
  40820f:	41 5c                	pop    %r12
  408211:	41 5d                	pop    %r13
  408213:	41 5e                	pop    %r14
  408215:	41 5f                	pop    %r15
  408217:	5d                   	pop    %rbp
  408218:	c3                   	retq   
  408219:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408220:	80 3d 9a 2e 21 00 00 	cmpb   $0x0,0x212e9a(%rip)        # 61b0c1 <stderr@@GLIBC_2.2.5+0xa71>
  408227:	0f 85 35 fd ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  40822d:	41 83 fd 03          	cmp    $0x3,%r13d
  408231:	0f 84 d9 06 00 00    	je     408910 <__sprintf_chk@plt+0x6080>
  408237:	0f b6 05 d6 2e 21 00 	movzbl 0x212ed6(%rip),%eax        # 61b114 <stderr@@GLIBC_2.2.5+0xac4>
  40823e:	84 c0                	test   %al,%al
  408240:	0f 84 d2 05 00 00    	je     408818 <__sprintf_chk@plt+0x5f88>
  408246:	45 85 ed             	test   %r13d,%r13d
  408249:	40 0f 94 c6          	sete   %sil
  40824d:	74 06                	je     408255 <__sprintf_chk@plt+0x59c5>
  40824f:	41 83 fd 06          	cmp    $0x6,%r13d
  408253:	75 2b                	jne    408280 <__sprintf_chk@plt+0x59f0>
  408255:	8b 0d b5 2e 21 00    	mov    0x212eb5(%rip),%ecx        # 61b110 <stderr@@GLIBC_2.2.5+0xac0>
  40825b:	83 f9 05             	cmp    $0x5,%ecx
  40825e:	0f 84 6c 09 00 00    	je     408bd0 <__sprintf_chk@plt+0x6340>
  408264:	80 3d 2d 2f 21 00 00 	cmpb   $0x0,0x212f2d(%rip)        # 61b198 <stderr@@GLIBC_2.2.5+0xb48>
  40826b:	0f 85 f7 fc ff ff    	jne    407f68 <__sprintf_chk@plt+0x56d8>
  408271:	80 3d 9d 2e 21 00 00 	cmpb   $0x0,0x212e9d(%rip)        # 61b115 <stderr@@GLIBC_2.2.5+0xac5>
  408278:	0f 85 ea fc ff ff    	jne    407f68 <__sprintf_chk@plt+0x56d8>
  40827e:	66 90                	xchg   %ax,%ax
  408280:	84 c0                	test   %al,%al
  408282:	0f 85 da fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  408288:	80 3d 31 2e 21 00 00 	cmpb   $0x0,0x212e31(%rip)        # 61b0c0 <stderr@@GLIBC_2.2.5+0xa70>
  40828f:	0f 84 90 05 00 00    	je     408825 <__sprintf_chk@plt+0x5f95>
  408295:	40 84 f6             	test   %sil,%sil
  408298:	0f 85 c4 fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  40829e:	45 31 e4             	xor    %r12d,%r12d
  4082a1:	41 83 fd 05          	cmp    $0x5,%r13d
  4082a5:	0f 85 36 ff ff ff    	jne    4081e1 <__sprintf_chk@plt+0x5951>
  4082ab:	83 3d 7a 2e 21 00 03 	cmpl   $0x3,0x212e7a(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  4082b2:	0f 84 aa fc ff ff    	je     407f62 <__sprintf_chk@plt+0x56d2>
  4082b8:	80 3d 6a 2e 21 00 00 	cmpb   $0x0,0x212e6a(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  4082bf:	0f 84 1c ff ff ff    	je     4081e1 <__sprintf_chk@plt+0x5951>
  4082c5:	bf 0e 00 00 00       	mov    $0xe,%edi
  4082ca:	48 89 95 70 fc ff ff 	mov    %rdx,-0x390(%rbp)
  4082d1:	e8 fa c9 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4082d6:	84 c0                	test   %al,%al
  4082d8:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  4082df:	0f 85 7d fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  4082e5:	bf 10 00 00 00       	mov    $0x10,%edi
  4082ea:	e8 e1 c9 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4082ef:	84 c0                	test   %al,%al
  4082f1:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  4082f8:	0f 85 64 fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  4082fe:	bf 11 00 00 00       	mov    $0x11,%edi
  408303:	e8 c8 c9 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  408308:	84 c0                	test   %al,%al
  40830a:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  408311:	0f 85 4b fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  408317:	bf 15 00 00 00       	mov    $0x15,%edi
  40831c:	e8 af c9 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  408321:	84 c0                	test   %al,%al
  408323:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  40832a:	0f 85 32 fc ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  408330:	e9 ac fe ff ff       	jmpq   4081e1 <__sprintf_chk@plt+0x5951>
  408335:	0f 1f 00             	nopl   (%rax)
  408338:	c6 07 00             	movb   $0x0,(%rdi)
  40833b:	e9 06 fc ff ff       	jmpq   407f46 <__sprintf_chk@plt+0x56b6>
  408340:	45 31 c0             	xor    %r8d,%r8d
  408343:	48 83 c7 02          	add    $0x2,%rdi
  408347:	66 44 89 47 fe       	mov    %r8w,-0x2(%rdi)
  40834c:	83 e6 01             	and    $0x1,%esi
  40834f:	0f 84 f1 fb ff ff    	je     407f46 <__sprintf_chk@plt+0x56b6>
  408355:	eb e1                	jmp    408338 <__sprintf_chk@plt+0x5aa8>
  408357:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40835e:	00 00 
  408360:	c7 07 00 00 00 00    	movl   $0x0,(%rdi)
  408366:	48 83 c7 04          	add    $0x4,%rdi
  40836a:	40 f6 c6 02          	test   $0x2,%sil
  40836e:	0f 84 c9 fb ff ff    	je     407f3d <__sprintf_chk@plt+0x56ad>
  408374:	eb ca                	jmp    408340 <__sprintf_chk@plt+0x5ab0>
  408376:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40837d:	00 00 00 
  408380:	80 3d a2 2d 21 00 00 	cmpb   $0x0,0x212da2(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  408387:	0f 84 b2 fc ff ff    	je     40803f <__sprintf_chk@plt+0x57af>
  40838d:	bf 15 00 00 00       	mov    $0x15,%edi
  408392:	89 b5 78 fc ff ff    	mov    %esi,-0x388(%rbp)
  408398:	89 95 70 fc ff ff    	mov    %edx,-0x390(%rbp)
  40839e:	e8 2d c9 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  4083a3:	84 c0                	test   %al,%al
  4083a5:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  4083ab:	8b b5 78 fc ff ff    	mov    -0x388(%rbp),%esi
  4083b1:	0f 84 88 fc ff ff    	je     40803f <__sprintf_chk@plt+0x57af>
  4083b7:	4d 8b 6e 10          	mov    0x10(%r14),%r13
  4083bb:	4c 3b 2d b6 22 21 00 	cmp    0x2122b6(%rip),%r13        # 61a678 <stderr@@GLIBC_2.2.5+0x28>
  4083c2:	0f 84 57 08 00 00    	je     408c1f <__sprintf_chk@plt+0x638f>
  4083c8:	89 b5 78 fc ff ff    	mov    %esi,-0x388(%rbp)
  4083ce:	89 95 70 fc ff ff    	mov    %edx,-0x390(%rbp)
  4083d4:	e8 57 9e ff ff       	callq  402230 <__errno_location@plt>
  4083d9:	8b b5 78 fc ff ff    	mov    -0x388(%rbp),%esi
  4083df:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  4083e5:	c7 00 5f 00 00 00    	movl   $0x5f,(%rax)
  4083eb:	4c 89 2d 86 22 21 00 	mov    %r13,0x212286(%rip)        # 61a678 <stderr@@GLIBC_2.2.5+0x28>
  4083f2:	41 c6 86 b8 00 00 00 	movb   $0x0,0xb8(%r14)
  4083f9:	00 
  4083fa:	e9 40 fc ff ff       	jmpq   40803f <__sprintf_chk@plt+0x57af>
  4083ff:	90                   	nop
  408400:	48 b8 aa aa aa aa aa 	movabs $0xaaaaaaaaaaaaaa,%rax
  408407:	aa aa 00 
  40840a:	48 8b 3d af 2d 21 00 	mov    0x212daf(%rip),%rdi        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  408411:	48 39 c1             	cmp    %rax,%rcx
  408414:	0f 87 61 08 00 00    	ja     408c7b <__sprintf_chk@plt+0x63eb>
  40841a:	48 8d 34 49          	lea    (%rcx,%rcx,2),%rsi
  40841e:	48 89 95 70 fc ff ff 	mov    %rdx,-0x390(%rbp)
  408425:	48 c1 e6 07          	shl    $0x7,%rsi
  408429:	e8 62 88 00 00       	callq  410c90 <__sprintf_chk@plt+0xe400>
  40842e:	48 d1 25 83 2d 21 00 	shlq   0x212d83(%rip)        # 61b1b8 <stderr@@GLIBC_2.2.5+0xb68>
  408435:	48 89 05 84 2d 21 00 	mov    %rax,0x212d84(%rip)        # 61b1c0 <stderr@@GLIBC_2.2.5+0xb70>
  40843c:	49 89 c1             	mov    %rax,%r9
  40843f:	48 8b 0d 6a 2d 21 00 	mov    0x212d6a(%rip),%rcx        # 61b1b0 <stderr@@GLIBC_2.2.5+0xb60>
  408446:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  40844d:	e9 9b fa ff ff       	jmpq   407eed <__sprintf_chk@plt+0x565d>
  408452:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408458:	e8 e3 93 00 00       	callq  411840 <__sprintf_chk@plt+0xefb0>
  40845d:	85 c0                	test   %eax,%eax
  40845f:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  408465:	0f 89 28 fc ff ff    	jns    408093 <__sprintf_chk@plt+0x5803>
  40846b:	89 95 70 fc ff ff    	mov    %edx,-0x390(%rbp)
  408471:	e8 ba 9d ff ff       	callq  402230 <__errno_location@plt>
  408476:	8b 00                	mov    (%rax),%eax
  408478:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  40847e:	83 f8 16             	cmp    $0x16,%eax
  408481:	74 09                	je     40848c <__sprintf_chk@plt+0x5bfc>
  408483:	83 f8 26             	cmp    $0x26,%eax
  408486:	0f 85 34 05 00 00    	jne    4089c0 <__sprintf_chk@plt+0x6130>
  40848c:	49 8b 4e 10          	mov    0x10(%r14),%rcx
  408490:	48 89 0d d9 21 21 00 	mov    %rcx,0x2121d9(%rip)        # 61a670 <stderr@@GLIBC_2.2.5+0x20>
  408497:	83 f8 5f             	cmp    $0x5f,%eax
  40849a:	49 c7 86 a8 00 00 00 	movq   $0x61a56a,0xa8(%r14)
  4084a1:	6a a5 61 00 
  4084a5:	0f 84 25 05 00 00    	je     4089d0 <__sprintf_chk@plt+0x6140>
  4084ab:	83 f8 3d             	cmp    $0x3d,%eax
  4084ae:	0f 84 1c 05 00 00    	je     4089d0 <__sprintf_chk@plt+0x6140>
  4084b4:	41 c7 86 b4 00 00 00 	movl   $0x0,0xb4(%r14)
  4084bb:	00 00 00 00 
  4084bf:	4c 89 e7             	mov    %r12,%rdi
  4084c2:	e8 e9 65 00 00       	callq  40eab0 <__sprintf_chk@plt+0xc220>
  4084c7:	49 89 c5             	mov    %rax,%r13
  4084ca:	e8 61 9d ff ff       	callq  402230 <__errno_location@plt>
  4084cf:	8b 30                	mov    (%rax),%esi
  4084d1:	31 ff                	xor    %edi,%edi
  4084d3:	31 c0                	xor    %eax,%eax
  4084d5:	4c 89 e9             	mov    %r13,%rcx
  4084d8:	ba 54 5e 41 00       	mov    $0x415e54,%edx
  4084dd:	e8 8e a2 ff ff       	callq  402770 <error@plt>
  4084e2:	41 8b 46 28          	mov    0x28(%r14),%eax
  4084e6:	25 00 f0 00 00       	and    $0xf000,%eax
  4084eb:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  4084f0:	0f 85 0d fc ff ff    	jne    408103 <__sprintf_chk@plt+0x5873>
  4084f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4084fd:	00 00 00 
  408500:	44 8b 2d 49 2c 21 00 	mov    0x212c49(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  408507:	45 85 ed             	test   %r13d,%r13d
  40850a:	74 0d                	je     408519 <__sprintf_chk@plt+0x5c89>
  40850c:	80 3d 02 2c 21 00 00 	cmpb   $0x0,0x212c02(%rip)        # 61b115 <stderr@@GLIBC_2.2.5+0xac5>
  408513:	0f 84 11 01 00 00    	je     40862a <__sprintf_chk@plt+0x5d9a>
  408519:	49 8b 76 40          	mov    0x40(%r14),%rsi
  40851d:	4c 89 e7             	mov    %r12,%rdi
  408520:	e8 fb 17 00 00       	callq  409d20 <__sprintf_chk@plt+0x7490>
  408525:	48 85 c0             	test   %rax,%rax
  408528:	49 89 c5             	mov    %rax,%r13
  40852b:	49 89 46 08          	mov    %rax,0x8(%r14)
  40852f:	0f 84 a5 06 00 00    	je     408bda <__sprintf_chk@plt+0x634a>
  408535:	41 80 7d 00 2f       	cmpb   $0x2f,0x0(%r13)
  40853a:	0f 84 cf 06 00 00    	je     408c0f <__sprintf_chk@plt+0x637f>
  408540:	4c 89 e7             	mov    %r12,%rdi
  408543:	e8 68 1d 00 00       	callq  40a2b0 <__sprintf_chk@plt+0x7a20>
  408548:	48 85 c0             	test   %rax,%rax
  40854b:	49 89 c7             	mov    %rax,%r15
  40854e:	4c 89 ef             	mov    %r13,%rdi
  408551:	0f 84 41 06 00 00    	je     408b98 <__sprintf_chk@plt+0x6308>
  408557:	e8 24 9e ff ff       	callq  402380 <strlen@plt>
  40855c:	49 8d 7c 07 02       	lea    0x2(%r15,%rax,1),%rdi
  408561:	e8 da 86 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  408566:	48 89 c1             	mov    %rax,%rcx
  408569:	31 c0                	xor    %eax,%eax
  40856b:	43 80 7c 3c ff 2f    	cmpb   $0x2f,-0x1(%r12,%r15,1)
  408571:	48 89 cf             	mov    %rcx,%rdi
  408574:	4c 89 e6             	mov    %r12,%rsi
  408577:	48 89 8d 70 fc ff ff 	mov    %rcx,-0x390(%rbp)
  40857e:	0f 95 c0             	setne  %al
  408581:	49 01 c7             	add    %rax,%r15
  408584:	4c 89 fa             	mov    %r15,%rdx
  408587:	e8 64 a1 ff ff       	callq  4026f0 <stpncpy@plt>
  40858c:	4c 89 ee             	mov    %r13,%rsi
  40858f:	48 89 c7             	mov    %rax,%rdi
  408592:	e8 c9 9c ff ff       	callq  402260 <strcpy@plt>
  408597:	48 8b 8d 70 fc ff ff 	mov    -0x390(%rbp),%rcx
  40859e:	49 89 cd             	mov    %rcx,%r13
  4085a1:	4d 85 ed             	test   %r13,%r13
  4085a4:	74 61                	je     408607 <__sprintf_chk@plt+0x5d77>
  4085a6:	83 3d 7f 2b 21 00 01 	cmpl   $0x1,0x212b7f(%rip)        # 61b12c <stderr@@GLIBC_2.2.5+0xadc>
  4085ad:	0f 86 cd 05 00 00    	jbe    408b80 <__sprintf_chk@plt+0x62f0>
  4085b3:	48 8d 95 80 fc ff ff 	lea    -0x380(%rbp),%rdx
  4085ba:	4c 89 ee             	mov    %r13,%rsi
  4085bd:	bf 01 00 00 00       	mov    $0x1,%edi
  4085c2:	e8 49 a0 ff ff       	callq  402610 <__xstat@plt>
  4085c7:	85 c0                	test   %eax,%eax
  4085c9:	75 3c                	jne    408607 <__sprintf_chk@plt+0x5d77>
  4085cb:	80 bd 7c fc ff ff 00 	cmpb   $0x0,-0x384(%rbp)
  4085d2:	41 c6 86 b1 00 00 00 	movb   $0x1,0xb1(%r14)
  4085d9:	01 
  4085da:	8b 85 98 fc ff ff    	mov    -0x368(%rbp),%eax
  4085e0:	74 1e                	je     408600 <__sprintf_chk@plt+0x5d70>
  4085e2:	8b 15 68 2b 21 00    	mov    0x212b68(%rip),%edx        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4085e8:	85 d2                	test   %edx,%edx
  4085ea:	74 14                	je     408600 <__sprintf_chk@plt+0x5d70>
  4085ec:	89 c2                	mov    %eax,%edx
  4085ee:	81 e2 00 f0 00 00    	and    $0xf000,%edx
  4085f4:	81 fa 00 40 00 00    	cmp    $0x4000,%edx
  4085fa:	74 0b                	je     408607 <__sprintf_chk@plt+0x5d77>
  4085fc:	0f 1f 40 00          	nopl   0x0(%rax)
  408600:	41 89 86 a4 00 00 00 	mov    %eax,0xa4(%r14)
  408607:	4c 89 ef             	mov    %r13,%rdi
  40860a:	e8 e1 9b ff ff       	callq  4021f0 <free@plt>
  40860f:	41 8b 46 28          	mov    0x28(%r14),%eax
  408613:	25 00 f0 00 00       	and    $0xf000,%eax
  408618:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  40861d:	0f 85 e0 fa ff ff    	jne    408103 <__sprintf_chk@plt+0x5873>
  408623:	44 8b 2d 26 2b 21 00 	mov    0x212b26(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  40862a:	41 c7 86 a0 00 00 00 	movl   $0x6,0xa0(%r14)
  408631:	06 00 00 00 
  408635:	e9 e6 fa ff ff       	jmpq   408120 <__sprintf_chk@plt+0x5890>
  40863a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408640:	49 8b 7e 18          	mov    0x18(%r14),%rdi
  408644:	48 8d b5 30 fd ff ff 	lea    -0x2d0(%rbp),%rsi
  40864b:	e8 20 47 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  408650:	48 89 c7             	mov    %rax,%rdi
  408653:	e8 28 9d ff ff       	callq  402380 <strlen@plt>
  408658:	3b 05 1a 2b 21 00    	cmp    0x212b1a(%rip),%eax        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  40865e:	0f 8e 7d fb ff ff    	jle    4081e1 <__sprintf_chk@plt+0x5951>
  408664:	89 05 0e 2b 21 00    	mov    %eax,0x212b0e(%rip)        # 61b178 <stderr@@GLIBC_2.2.5+0xb28>
  40866a:	e9 72 fb ff ff       	jmpq   4081e1 <__sprintf_chk@plt+0x5951>
  40866f:	90                   	nop
  408670:	48 89 df             	mov    %rbx,%rdi
  408673:	89 8d 6c fc ff ff    	mov    %ecx,-0x394(%rbp)
  408679:	89 b5 78 fc ff ff    	mov    %esi,-0x388(%rbp)
  40867f:	48 89 95 70 fc ff ff 	mov    %rdx,-0x390(%rbp)
  408686:	e8 f5 9c ff ff       	callq  402380 <strlen@plt>
  40868b:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  408692:	49 89 c4             	mov    %rax,%r12
  408695:	48 89 d7             	mov    %rdx,%rdi
  408698:	e8 e3 9c ff ff       	callq  402380 <strlen@plt>
  40869d:	49 8d 44 04 20       	lea    0x20(%r12,%rax,1),%rax
  4086a2:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  4086a9:	8b b5 78 fc ff ff    	mov    -0x388(%rbp),%esi
  4086af:	8b 8d 6c fc ff ff    	mov    -0x394(%rbp),%ecx
  4086b5:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
  4086b9:	48 29 c4             	sub    %rax,%rsp
  4086bc:	48 8d 44 24 0f       	lea    0xf(%rsp),%rax
  4086c1:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
  4086c5:	41 80 ff 2e          	cmp    $0x2e,%r15b
  4086c9:	49 89 c4             	mov    %rax,%r12
  4086cc:	0f 84 5e 04 00 00    	je     408b30 <__sprintf_chk@plt+0x62a0>
  4086d2:	48 89 d6             	mov    %rdx,%rsi
  4086d5:	0f 1f 00             	nopl   (%rax)
  4086d8:	48 83 c0 01          	add    $0x1,%rax
  4086dc:	48 83 c6 01          	add    $0x1,%rsi
  4086e0:	44 88 78 ff          	mov    %r15b,-0x1(%rax)
  4086e4:	44 0f b6 3e          	movzbl (%rsi),%r15d
  4086e8:	45 84 ff             	test   %r15b,%r15b
  4086eb:	75 eb                	jne    4086d8 <__sprintf_chk@plt+0x5e48>
  4086ed:	48 39 f2             	cmp    %rsi,%rdx
  4086f0:	48 89 c7             	mov    %rax,%rdi
  4086f3:	73 0d                	jae    408702 <__sprintf_chk@plt+0x5e72>
  4086f5:	80 7e ff 2f          	cmpb   $0x2f,-0x1(%rsi)
  4086f9:	74 07                	je     408702 <__sprintf_chk@plt+0x5e72>
  4086fb:	48 83 c0 01          	add    $0x1,%rax
  4086ff:	c6 07 2f             	movb   $0x2f,(%rdi)
  408702:	0f b6 33             	movzbl (%rbx),%esi
  408705:	40 84 f6             	test   %sil,%sil
  408708:	74 1a                	je     408724 <__sprintf_chk@plt+0x5e94>
  40870a:	48 89 da             	mov    %rbx,%rdx
  40870d:	0f 1f 00             	nopl   (%rax)
  408710:	48 83 c0 01          	add    $0x1,%rax
  408714:	48 83 c2 01          	add    $0x1,%rdx
  408718:	40 88 70 ff          	mov    %sil,-0x1(%rax)
  40871c:	0f b6 32             	movzbl (%rdx),%esi
  40871f:	40 84 f6             	test   %sil,%sil
  408722:	75 ec                	jne    408710 <__sprintf_chk@plt+0x5e80>
  408724:	c6 00 00             	movb   $0x0,(%rax)
  408727:	e9 55 f8 ff ff       	jmpq   407f81 <__sprintf_chk@plt+0x56f1>
  40872c:	0f 1f 40 00          	nopl   0x0(%rax)
  408730:	45 85 ed             	test   %r13d,%r13d
  408733:	89 05 33 2a 21 00    	mov    %eax,0x212a33(%rip)        # 61b16c <stderr@@GLIBC_2.2.5+0xb1c>
  408739:	0f 85 93 fa ff ff    	jne    4081d2 <__sprintf_chk@plt+0x5942>
  40873f:	49 8b 7e 20          	mov    0x20(%r14),%rdi
  408743:	48 8d b5 10 fd ff ff 	lea    -0x2f0(%rbp),%rsi
  40874a:	e8 21 46 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  40874f:	48 89 c7             	mov    %rax,%rdi
  408752:	e8 29 9c ff ff       	callq  402380 <strlen@plt>
  408757:	3b 05 13 2a 21 00    	cmp    0x212a13(%rip),%eax        # 61b170 <stderr@@GLIBC_2.2.5+0xb20>
  40875d:	7e 06                	jle    408765 <__sprintf_chk@plt+0x5ed5>
  40875f:	89 05 0b 2a 21 00    	mov    %eax,0x212a0b(%rip)        # 61b170 <stderr@@GLIBC_2.2.5+0xb20>
  408765:	41 8b 46 28          	mov    0x28(%r14),%eax
  408769:	25 00 b0 00 00       	and    $0xb000,%eax
  40876e:	3d 00 20 00 00       	cmp    $0x2000,%eax
  408773:	0f 85 b7 02 00 00    	jne    408a30 <__sprintf_chk@plt+0x61a0>
  408779:	49 8b 46 38          	mov    0x38(%r14),%rax
  40877d:	4c 8d bd 30 fd ff ff 	lea    -0x2d0(%rbp),%r15
  408784:	4c 89 fe             	mov    %r15,%rsi
  408787:	48 89 c7             	mov    %rax,%rdi
  40878a:	48 c1 e8 08          	shr    $0x8,%rax
  40878e:	48 c1 ef 20          	shr    $0x20,%rdi
  408792:	25 ff 0f 00 00       	and    $0xfff,%eax
  408797:	81 e7 00 f0 ff ff    	and    $0xfffff000,%edi
  40879d:	09 c7                	or     %eax,%edi
  40879f:	e8 cc 45 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  4087a4:	48 89 c7             	mov    %rax,%rdi
  4087a7:	e8 d4 9b ff ff       	callq  402380 <strlen@plt>
  4087ac:	3b 05 aa 29 21 00    	cmp    0x2129aa(%rip),%eax        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  4087b2:	7e 06                	jle    4087ba <__sprintf_chk@plt+0x5f2a>
  4087b4:	89 05 a2 29 21 00    	mov    %eax,0x2129a2(%rip)        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  4087ba:	49 8b 46 38          	mov    0x38(%r14),%rax
  4087be:	4c 89 fe             	mov    %r15,%rsi
  4087c1:	48 89 c7             	mov    %rax,%rdi
  4087c4:	0f b6 c0             	movzbl %al,%eax
  4087c7:	48 c1 ef 0c          	shr    $0xc,%rdi
  4087cb:	40 80 e7 00          	and    $0x0,%dil
  4087cf:	09 c7                	or     %eax,%edi
  4087d1:	e8 9a 45 00 00       	callq  40cd70 <__sprintf_chk@plt+0xa4e0>
  4087d6:	48 89 c7             	mov    %rax,%rdi
  4087d9:	e8 a2 9b ff ff       	callq  402380 <strlen@plt>
  4087de:	8b 15 74 29 21 00    	mov    0x212974(%rip),%edx        # 61b158 <stderr@@GLIBC_2.2.5+0xb08>
  4087e4:	39 d0                	cmp    %edx,%eax
  4087e6:	7e 08                	jle    4087f0 <__sprintf_chk@plt+0x5f60>
  4087e8:	89 05 6a 29 21 00    	mov    %eax,0x21296a(%rip)        # 61b158 <stderr@@GLIBC_2.2.5+0xb08>
  4087ee:	89 c2                	mov    %eax,%edx
  4087f0:	8b 05 66 29 21 00    	mov    0x212966(%rip),%eax        # 61b15c <stderr@@GLIBC_2.2.5+0xb0c>
  4087f6:	8d 44 02 02          	lea    0x2(%rdx,%rax,1),%eax
  4087fa:	3b 05 54 29 21 00    	cmp    0x212954(%rip),%eax        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  408800:	0f 8e cc f9 ff ff    	jle    4081d2 <__sprintf_chk@plt+0x5942>
  408806:	89 05 48 29 21 00    	mov    %eax,0x212948(%rip)        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  40880c:	e9 c1 f9 ff ff       	jmpq   4081d2 <__sprintf_chk@plt+0x5942>
  408811:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408818:	80 3d a1 28 21 00 00 	cmpb   $0x0,0x2128a1(%rip)        # 61b0c0 <stderr@@GLIBC_2.2.5+0xa70>
  40881f:	0f 85 21 fa ff ff    	jne    408246 <__sprintf_chk@plt+0x59b6>
  408825:	45 31 e4             	xor    %r12d,%r12d
  408828:	e9 b4 f9 ff ff       	jmpq   4081e1 <__sprintf_chk@plt+0x5951>
  40882d:	0f 1f 00             	nopl   (%rax)
  408830:	48 8b 05 31 1e 21 00 	mov    0x211e31(%rip),%rax        # 61a668 <stderr@@GLIBC_2.2.5+0x18>
  408837:	49 39 46 10          	cmp    %rax,0x10(%r14)
  40883b:	0f 84 ff 02 00 00    	je     408b40 <__sprintf_chk@plt+0x62b0>
  408841:	89 95 78 fc ff ff    	mov    %edx,-0x388(%rbp)
  408847:	e8 e4 99 ff ff       	callq  402230 <__errno_location@plt>
  40884c:	4c 89 fe             	mov    %r15,%rsi
  40884f:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  408855:	4c 89 e7             	mov    %r12,%rdi
  408858:	48 89 85 70 fc ff ff 	mov    %rax,-0x390(%rbp)
  40885f:	e8 5c 14 00 00       	callq  409cc0 <__sprintf_chk@plt+0x7430>
  408864:	85 c0                	test   %eax,%eax
  408866:	b9 01 00 00 00       	mov    $0x1,%ecx
  40886b:	4c 8b 85 70 fc ff ff 	mov    -0x390(%rbp),%r8
  408872:	8b 95 78 fc ff ff    	mov    -0x388(%rbp),%edx
  408878:	0f 8f 3c f8 ff ff    	jg     4080ba <__sprintf_chk@plt+0x582a>
  40887e:	41 8b 10             	mov    (%r8),%edx
  408881:	83 fa 16             	cmp    $0x16,%edx
  408884:	0f 84 b1 03 00 00    	je     408c3b <__sprintf_chk@plt+0x63ab>
  40888a:	83 fa 26             	cmp    $0x26,%edx
  40888d:	0f 84 a8 03 00 00    	je     408c3b <__sprintf_chk@plt+0x63ab>
  408893:	83 fa 5f             	cmp    $0x5f,%edx
  408896:	0f 84 9f 03 00 00    	je     408c3b <__sprintf_chk@plt+0x63ab>
  40889c:	c1 e8 1f             	shr    $0x1f,%eax
  40889f:	89 c2                	mov    %eax,%edx
  4088a1:	e9 12 f8 ff ff       	jmpq   4080b8 <__sprintf_chk@plt+0x5828>
  4088a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4088ad:	00 00 00 
  4088b0:	4d 8d 7e 10          	lea    0x10(%r14),%r15
  4088b4:	4c 89 e6             	mov    %r12,%rsi
  4088b7:	bf 01 00 00 00       	mov    $0x1,%edi
  4088bc:	89 8d 70 fc ff ff    	mov    %ecx,-0x390(%rbp)
  4088c2:	4c 89 fa             	mov    %r15,%rdx
  4088c5:	e8 46 9d ff ff       	callq  402610 <__xstat@plt>
  4088ca:	8b 8d 70 fc ff ff    	mov    -0x390(%rbp),%ecx
  4088d0:	89 c2                	mov    %eax,%edx
  4088d2:	be 01 00 00 00       	mov    $0x1,%esi
  4088d7:	83 f9 03             	cmp    $0x3,%ecx
  4088da:	0f 84 35 f7 ff ff    	je     408015 <__sprintf_chk@plt+0x5785>
  4088e0:	85 c0                	test   %eax,%eax
  4088e2:	0f 88 76 02 00 00    	js     408b5e <__sprintf_chk@plt+0x62ce>
  4088e8:	41 8b 46 28          	mov    0x28(%r14),%eax
  4088ec:	25 00 f0 00 00       	and    $0xf000,%eax
  4088f1:	3d 00 40 00 00       	cmp    $0x4000,%eax
  4088f6:	0f 95 c0             	setne  %al
  4088f9:	84 c0                	test   %al,%al
  4088fb:	be 01 00 00 00       	mov    $0x1,%esi
  408900:	0f 84 0f f7 ff ff    	je     408015 <__sprintf_chk@plt+0x5785>
  408906:	e9 f6 f6 ff ff       	jmpq   408001 <__sprintf_chk@plt+0x5771>
  40890b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  408910:	80 3d 12 28 21 00 00 	cmpb   $0x0,0x212812(%rip)        # 61b129 <stderr@@GLIBC_2.2.5+0xad9>
  408917:	0f 84 1a f9 ff ff    	je     408237 <__sprintf_chk@plt+0x59a7>
  40891d:	bf 13 00 00 00       	mov    $0x13,%edi
  408922:	48 89 95 70 fc ff ff 	mov    %rdx,-0x390(%rbp)
  408929:	e8 a2 c3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  40892e:	84 c0                	test   %al,%al
  408930:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  408937:	0f 85 25 f6 ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  40893d:	bf 12 00 00 00       	mov    $0x12,%edi
  408942:	e8 89 c3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  408947:	84 c0                	test   %al,%al
  408949:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  408950:	0f 85 0c f6 ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  408956:	bf 14 00 00 00       	mov    $0x14,%edi
  40895b:	e8 70 c3 ff ff       	callq  404cd0 <__sprintf_chk@plt+0x2440>
  408960:	84 c0                	test   %al,%al
  408962:	48 8b 95 70 fc ff ff 	mov    -0x390(%rbp),%rdx
  408969:	0f 85 f3 f5 ff ff    	jne    407f62 <__sprintf_chk@plt+0x56d2>
  40896f:	e9 c3 f8 ff ff       	jmpq   408237 <__sprintf_chk@plt+0x59a7>
  408974:	0f 1f 40 00          	nopl   0x0(%rax)
  408978:	80 bd 7c fc ff ff 00 	cmpb   $0x0,-0x384(%rbp)
  40897f:	74 27                	je     4089a8 <__sprintf_chk@plt+0x6118>
  408981:	80 3d 85 27 21 00 00 	cmpb   $0x0,0x212785(%rip)        # 61b10d <stderr@@GLIBC_2.2.5+0xabd>
  408988:	75 1e                	jne    4089a8 <__sprintf_chk@plt+0x6118>
  40898a:	41 c7 86 a0 00 00 00 	movl   $0x9,0xa0(%r14)
  408991:	09 00 00 00 
  408995:	44 8b 2d b4 27 21 00 	mov    0x2127b4(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  40899c:	e9 7f f7 ff ff       	jmpq   408120 <__sprintf_chk@plt+0x5890>
  4089a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4089a8:	41 c7 86 a0 00 00 00 	movl   $0x3,0xa0(%r14)
  4089af:	03 00 00 00 
  4089b3:	44 8b 2d 96 27 21 00 	mov    0x212796(%rip),%r13d        # 61b150 <stderr@@GLIBC_2.2.5+0xb00>
  4089ba:	e9 61 f7 ff ff       	jmpq   408120 <__sprintf_chk@plt+0x5890>
  4089bf:	90                   	nop
  4089c0:	83 f8 5f             	cmp    $0x5f,%eax
  4089c3:	0f 85 ce fa ff ff    	jne    408497 <__sprintf_chk@plt+0x5c07>
  4089c9:	e9 be fa ff ff       	jmpq   40848c <__sprintf_chk@plt+0x5bfc>
  4089ce:	66 90                	xchg   %ax,%ax
  4089d0:	45 31 ed             	xor    %r13d,%r13d
  4089d3:	e9 d2 f6 ff ff       	jmpq   4080aa <__sprintf_chk@plt+0x581a>
  4089d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4089df:	00 
  4089e0:	45 31 c9             	xor    %r9d,%r9d
  4089e3:	48 83 c7 02          	add    $0x2,%rdi
  4089e7:	83 ee 02             	sub    $0x2,%esi
  4089ea:	66 44 89 4f fe       	mov    %r9w,-0x2(%rdi)
  4089ef:	40 f6 c7 04          	test   $0x4,%dil
  4089f3:	0f 84 26 f5 ff ff    	je     407f1f <__sprintf_chk@plt+0x568f>
  4089f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408a00:	c7 07 00 00 00 00    	movl   $0x0,(%rdi)
  408a06:	83 ee 04             	sub    $0x4,%esi
  408a09:	48 83 c7 04          	add    $0x4,%rdi
  408a0d:	e9 0d f5 ff ff       	jmpq   407f1f <__sprintf_chk@plt+0x568f>
  408a12:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408a18:	41 c6 06 00          	movb   $0x0,(%r14)
  408a1c:	49 8d 7e 01          	lea    0x1(%r14),%rdi
  408a20:	40 b6 bf             	mov    $0xbf,%sil
  408a23:	e9 e3 f4 ff ff       	jmpq   407f0b <__sprintf_chk@plt+0x567b>
  408a28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  408a2f:	00 
  408a30:	49 8b 7e 40          	mov    0x40(%r14),%rdi
  408a34:	4c 8b 05 25 1b 21 00 	mov    0x211b25(%rip),%r8        # 61a560 <_fini@@Base+0x208664>
  408a3b:	48 8d b5 30 fd ff ff 	lea    -0x2d0(%rbp),%rsi
  408a42:	8b 15 ec 26 21 00    	mov    0x2126ec(%rip),%edx        # 61b134 <stderr@@GLIBC_2.2.5+0xae4>
  408a48:	b9 01 00 00 00       	mov    $0x1,%ecx
  408a4d:	e8 1e 33 00 00       	callq  40bd70 <__sprintf_chk@plt+0x94e0>
  408a52:	31 f6                	xor    %esi,%esi
  408a54:	48 89 c7             	mov    %rax,%rdi
  408a57:	e8 c4 49 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  408a5c:	3b 05 f2 26 21 00    	cmp    0x2126f2(%rip),%eax        # 61b154 <stderr@@GLIBC_2.2.5+0xb04>
  408a62:	0f 8e 6a f7 ff ff    	jle    4081d2 <__sprintf_chk@plt+0x5942>
  408a68:	e9 99 fd ff ff       	jmpq   408806 <__sprintf_chk@plt+0x5f76>
  408a6d:	0f 1f 00             	nopl   (%rax)
  408a70:	41 8b 7e 2c          	mov    0x2c(%r14),%edi
  408a74:	e8 37 d7 ff ff       	callq  4061b0 <__sprintf_chk@plt+0x3920>
  408a79:	3b 05 e1 26 21 00    	cmp    0x2126e1(%rip),%eax        # 61b160 <stderr@@GLIBC_2.2.5+0xb10>
  408a7f:	0f 8e 1c f7 ff ff    	jle    4081a1 <__sprintf_chk@plt+0x5911>
  408a85:	89 05 d5 26 21 00    	mov    %eax,0x2126d5(%rip)        # 61b160 <stderr@@GLIBC_2.2.5+0xb10>
  408a8b:	e9 11 f7 ff ff       	jmpq   4081a1 <__sprintf_chk@plt+0x5911>
  408a90:	80 3d ae 26 21 00 00 	cmpb   $0x0,0x2126ae(%rip)        # 61b145 <stderr@@GLIBC_2.2.5+0xaf5>
  408a97:	45 8b 6e 30          	mov    0x30(%r14),%r13d
  408a9b:	0f 84 07 01 00 00    	je     408ba8 <__sprintf_chk@plt+0x6318>
  408aa1:	ba 15 00 00 00       	mov    $0x15,%edx
  408aa6:	45 89 e8             	mov    %r13d,%r8d
  408aa9:	b9 5a 37 41 00       	mov    $0x41375a,%ecx
  408aae:	be 01 00 00 00       	mov    $0x1,%esi
  408ab3:	4c 89 ff             	mov    %r15,%rdi
  408ab6:	31 c0                	xor    %eax,%eax
  408ab8:	e8 d3 9d ff ff       	callq  402890 <__sprintf_chk@plt>
  408abd:	4c 89 fa             	mov    %r15,%rdx
  408ac0:	8b 0a                	mov    (%rdx),%ecx
  408ac2:	48 83 c2 04          	add    $0x4,%rdx
  408ac6:	8d 81 ff fe fe fe    	lea    -0x1010101(%rcx),%eax
  408acc:	f7 d1                	not    %ecx
  408ace:	21 c8                	and    %ecx,%eax
  408ad0:	25 80 80 80 80       	and    $0x80808080,%eax
  408ad5:	74 e9                	je     408ac0 <__sprintf_chk@plt+0x6230>
  408ad7:	89 c1                	mov    %eax,%ecx
  408ad9:	c1 e9 10             	shr    $0x10,%ecx
  408adc:	a9 80 80 00 00       	test   $0x8080,%eax
  408ae1:	0f 44 c1             	cmove  %ecx,%eax
  408ae4:	48 8d 4a 02          	lea    0x2(%rdx),%rcx
  408ae8:	48 0f 44 d1          	cmove  %rcx,%rdx
  408aec:	00 c0                	add    %al,%al
  408aee:	48 83 da 03          	sbb    $0x3,%rdx
  408af2:	44 29 fa             	sub    %r15d,%edx
  408af5:	39 15 69 26 21 00    	cmp    %edx,0x212669(%rip)        # 61b164 <stderr@@GLIBC_2.2.5+0xb14>
  408afb:	0f 8d 93 f6 ff ff    	jge    408194 <__sprintf_chk@plt+0x5904>
  408b01:	89 15 5d 26 21 00    	mov    %edx,0x21265d(%rip)        # 61b164 <stderr@@GLIBC_2.2.5+0xb14>
  408b07:	e9 88 f6 ff ff       	jmpq   408194 <__sprintf_chk@plt+0x5904>
  408b0c:	0f 1f 40 00          	nopl   0x0(%rax)
  408b10:	41 8b 7e 2c          	mov    0x2c(%r14),%edi
  408b14:	e8 97 d6 ff ff       	callq  4061b0 <__sprintf_chk@plt+0x3920>
  408b19:	3b 05 49 26 21 00    	cmp    0x212649(%rip),%eax        # 61b168 <stderr@@GLIBC_2.2.5+0xb18>
  408b1f:	0f 8e 62 f6 ff ff    	jle    408187 <__sprintf_chk@plt+0x58f7>
  408b25:	89 05 3d 26 21 00    	mov    %eax,0x21263d(%rip)        # 61b168 <stderr@@GLIBC_2.2.5+0xb18>
  408b2b:	e9 57 f6 ff ff       	jmpq   408187 <__sprintf_chk@plt+0x58f7>
  408b30:	80 7a 01 00          	cmpb   $0x0,0x1(%rdx)
  408b34:	0f 84 cb fb ff ff    	je     408705 <__sprintf_chk@plt+0x5e75>
  408b3a:	e9 93 fb ff ff       	jmpq   4086d2 <__sprintf_chk@plt+0x5e42>
  408b3f:	90                   	nop
  408b40:	89 95 70 fc ff ff    	mov    %edx,-0x390(%rbp)
  408b46:	e8 e5 96 ff ff       	callq  402230 <__errno_location@plt>
  408b4b:	31 c9                	xor    %ecx,%ecx
  408b4d:	c7 00 5f 00 00 00    	movl   $0x5f,(%rax)
  408b53:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  408b59:	e9 5c f5 ff ff       	jmpq   4080ba <__sprintf_chk@plt+0x582a>
  408b5e:	89 85 70 fc ff ff    	mov    %eax,-0x390(%rbp)
  408b64:	e8 c7 96 ff ff       	callq  402230 <__errno_location@plt>
  408b69:	83 38 02             	cmpl   $0x2,(%rax)
  408b6c:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  408b72:	0f 94 c0             	sete   %al
  408b75:	e9 7f fd ff ff       	jmpq   4088f9 <__sprintf_chk@plt+0x6069>
  408b7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408b80:	80 3d 8e 25 21 00 00 	cmpb   $0x0,0x21258e(%rip)        # 61b115 <stderr@@GLIBC_2.2.5+0xac5>
  408b87:	0f 84 7a fa ff ff    	je     408607 <__sprintf_chk@plt+0x5d77>
  408b8d:	e9 21 fa ff ff       	jmpq   4085b3 <__sprintf_chk@plt+0x5d23>
  408b92:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408b98:	e8 93 82 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  408b9d:	49 89 c5             	mov    %rax,%r13
  408ba0:	e9 fc f9 ff ff       	jmpq   4085a1 <__sprintf_chk@plt+0x5d11>
  408ba5:	0f 1f 00             	nopl   (%rax)
  408ba8:	44 89 ef             	mov    %r13d,%edi
  408bab:	e8 90 3f 00 00       	callq  40cb40 <__sprintf_chk@plt+0xa2b0>
  408bb0:	48 85 c0             	test   %rax,%rax
  408bb3:	48 89 c7             	mov    %rax,%rdi
  408bb6:	0f 84 e5 fe ff ff    	je     408aa1 <__sprintf_chk@plt+0x6211>
  408bbc:	31 f6                	xor    %esi,%esi
  408bbe:	e8 5d 48 00 00       	callq  40d420 <__sprintf_chk@plt+0xab90>
  408bc3:	31 d2                	xor    %edx,%edx
  408bc5:	85 c0                	test   %eax,%eax
  408bc7:	0f 49 d0             	cmovns %eax,%edx
  408bca:	e9 26 ff ff ff       	jmpq   408af5 <__sprintf_chk@plt+0x6265>
  408bcf:	90                   	nop
  408bd0:	b9 05 00 00 00       	mov    $0x5,%ecx
  408bd5:	e9 8e f3 ff ff       	jmpq   407f68 <__sprintf_chk@plt+0x56d8>
  408bda:	31 ff                	xor    %edi,%edi
  408bdc:	ba 05 00 00 00       	mov    $0x5,%edx
  408be1:	be 94 37 41 00       	mov    $0x413794,%esi
  408be6:	e8 75 97 ff ff       	callq  402360 <dcgettext@plt>
  408beb:	0f b6 bd 7c fc ff ff 	movzbl -0x384(%rbp),%edi
  408bf2:	4c 89 e2             	mov    %r12,%rdx
  408bf5:	48 89 c6             	mov    %rax,%rsi
  408bf8:	e8 13 cc ff ff       	callq  405810 <__sprintf_chk@plt+0x2f80>
  408bfd:	4d 8b 6e 08          	mov    0x8(%r14),%r13
  408c01:	4d 85 ed             	test   %r13,%r13
  408c04:	0f 85 2b f9 ff ff    	jne    408535 <__sprintf_chk@plt+0x5ca5>
  408c0a:	e9 f8 f9 ff ff       	jmpq   408607 <__sprintf_chk@plt+0x5d77>
  408c0f:	4c 89 ef             	mov    %r13,%rdi
  408c12:	e8 19 82 00 00       	callq  410e30 <__sprintf_chk@plt+0xe5a0>
  408c17:	49 89 c5             	mov    %rax,%r13
  408c1a:	e9 82 f9 ff ff       	jmpq   4085a1 <__sprintf_chk@plt+0x5d11>
  408c1f:	e8 0c 96 ff ff       	callq  402230 <__errno_location@plt>
  408c24:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  408c2a:	c7 00 5f 00 00 00    	movl   $0x5f,(%rax)
  408c30:	8b b5 78 fc ff ff    	mov    -0x388(%rbp),%esi
  408c36:	e9 b7 f7 ff ff       	jmpq   4083f2 <__sprintf_chk@plt+0x5b62>
  408c3b:	49 8b 56 10          	mov    0x10(%r14),%rdx
  408c3f:	c1 e8 1f             	shr    $0x1f,%eax
  408c42:	31 c9                	xor    %ecx,%ecx
  408c44:	48 89 15 1d 1a 21 00 	mov    %rdx,0x211a1d(%rip)        # 61a668 <stderr@@GLIBC_2.2.5+0x18>
  408c4b:	89 c2                	mov    %eax,%edx
  408c4d:	e9 68 f4 ff ff       	jmpq   4080ba <__sprintf_chk@plt+0x582a>
  408c52:	e8 d9 95 ff ff       	callq  402230 <__errno_location@plt>
  408c57:	45 31 ed             	xor    %r13d,%r13d
  408c5a:	c7 00 5f 00 00 00    	movl   $0x5f,(%rax)
  408c60:	8b 95 70 fc ff ff    	mov    -0x390(%rbp),%edx
  408c66:	49 c7 86 a8 00 00 00 	movq   $0x61a56a,0xa8(%r14)
  408c6d:	6a a5 61 00 
  408c71:	e9 34 f4 ff ff       	jmpq   4080aa <__sprintf_chk@plt+0x581a>
  408c76:	e8 25 97 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  408c7b:	e8 d0 81 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  408c80:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408c86:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408c8d:	83 f8 09             	cmp    $0x9,%eax
  408c90:	0f 94 c1             	sete   %cl
  408c93:	83 f8 03             	cmp    $0x3,%eax
  408c96:	0f 94 c0             	sete   %al
  408c99:	41 83 f8 09          	cmp    $0x9,%r8d
  408c9d:	0f 94 c2             	sete   %dl
  408ca0:	41 83 f8 03          	cmp    $0x3,%r8d
  408ca4:	41 0f 94 c0          	sete   %r8b
  408ca8:	44 09 c2             	or     %r8d,%edx
  408cab:	08 c8                	or     %cl,%al
  408cad:	75 39                	jne    408ce8 <__sprintf_chk@plt+0x6458>
  408caf:	84 c0                	test   %al,%al
  408cb1:	75 0d                	jne    408cc0 <__sprintf_chk@plt+0x6430>
  408cb3:	84 d2                	test   %dl,%dl
  408cb5:	b8 01 00 00 00       	mov    $0x1,%eax
  408cba:	74 04                	je     408cc0 <__sprintf_chk@plt+0x6430>
  408cbc:	f3 c3                	repz retq 
  408cbe:	66 90                	xchg   %ax,%ax
  408cc0:	48 8b 4e 68          	mov    0x68(%rsi),%rcx
  408cc4:	48 39 4f 68          	cmp    %rcx,0x68(%rdi)
  408cc8:	48 8b 47 70          	mov    0x70(%rdi),%rax
  408ccc:	48 8b 56 70          	mov    0x70(%rsi),%rdx
  408cd0:	7f 1e                	jg     408cf0 <__sprintf_chk@plt+0x6460>
  408cd2:	7c 2c                	jl     408d00 <__sprintf_chk@plt+0x6470>
  408cd4:	29 c2                	sub    %eax,%edx
  408cd6:	75 2e                	jne    408d06 <__sprintf_chk@plt+0x6476>
  408cd8:	48 8b 36             	mov    (%rsi),%rsi
  408cdb:	48 8b 3f             	mov    (%rdi),%rdi
  408cde:	e9 3d c3 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408ce3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  408ce8:	84 d2                	test   %dl,%dl
  408cea:	75 c3                	jne    408caf <__sprintf_chk@plt+0x641f>
  408cec:	0f 1f 40 00          	nopl   0x0(%rax)
  408cf0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408cf5:	c3                   	retq   
  408cf6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  408cfd:	00 00 00 
  408d00:	b8 01 00 00 00       	mov    $0x1,%eax
  408d05:	c3                   	retq   
  408d06:	89 d0                	mov    %edx,%eax
  408d08:	c3                   	retq   
  408d09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408d10:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408d16:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408d1d:	49 89 f1             	mov    %rsi,%r9
  408d20:	83 f8 09             	cmp    $0x9,%eax
  408d23:	0f 94 c1             	sete   %cl
  408d26:	83 f8 03             	cmp    $0x3,%eax
  408d29:	0f 94 c0             	sete   %al
  408d2c:	41 83 f8 09          	cmp    $0x9,%r8d
  408d30:	0f 94 c2             	sete   %dl
  408d33:	41 83 f8 03          	cmp    $0x3,%r8d
  408d37:	40 0f 94 c6          	sete   %sil
  408d3b:	09 f2                	or     %esi,%edx
  408d3d:	08 c8                	or     %cl,%al
  408d3f:	75 3f                	jne    408d80 <__sprintf_chk@plt+0x64f0>
  408d41:	84 c0                	test   %al,%al
  408d43:	75 13                	jne    408d58 <__sprintf_chk@plt+0x64c8>
  408d45:	84 d2                	test   %dl,%dl
  408d47:	b8 01 00 00 00       	mov    $0x1,%eax
  408d4c:	74 0a                	je     408d58 <__sprintf_chk@plt+0x64c8>
  408d4e:	66 90                	xchg   %ax,%ax
  408d50:	f3 c3                	repz retq 
  408d52:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408d58:	48 8b 4f 68          	mov    0x68(%rdi),%rcx
  408d5c:	49 39 49 68          	cmp    %rcx,0x68(%r9)
  408d60:	49 8b 41 70          	mov    0x70(%r9),%rax
  408d64:	48 8b 57 70          	mov    0x70(%rdi),%rdx
  408d68:	7f 1e                	jg     408d88 <__sprintf_chk@plt+0x64f8>
  408d6a:	7c 24                	jl     408d90 <__sprintf_chk@plt+0x6500>
  408d6c:	29 c2                	sub    %eax,%edx
  408d6e:	75 26                	jne    408d96 <__sprintf_chk@plt+0x6506>
  408d70:	48 8b 37             	mov    (%rdi),%rsi
  408d73:	49 8b 39             	mov    (%r9),%rdi
  408d76:	e9 a5 c2 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408d7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  408d80:	84 d2                	test   %dl,%dl
  408d82:	75 bd                	jne    408d41 <__sprintf_chk@plt+0x64b1>
  408d84:	0f 1f 40 00          	nopl   0x0(%rax)
  408d88:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408d8d:	c3                   	retq   
  408d8e:	66 90                	xchg   %ax,%ax
  408d90:	b8 01 00 00 00       	mov    $0x1,%eax
  408d95:	c3                   	retq   
  408d96:	89 d0                	mov    %edx,%eax
  408d98:	c3                   	retq   
  408d99:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408da0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408da6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408dad:	83 f8 09             	cmp    $0x9,%eax
  408db0:	0f 94 c1             	sete   %cl
  408db3:	83 f8 03             	cmp    $0x3,%eax
  408db6:	0f 94 c0             	sete   %al
  408db9:	41 83 f8 09          	cmp    $0x9,%r8d
  408dbd:	0f 94 c2             	sete   %dl
  408dc0:	41 83 f8 03          	cmp    $0x3,%r8d
  408dc4:	41 0f 94 c0          	sete   %r8b
  408dc8:	44 09 c2             	or     %r8d,%edx
  408dcb:	08 c8                	or     %cl,%al
  408dcd:	75 41                	jne    408e10 <__sprintf_chk@plt+0x6580>
  408dcf:	84 c0                	test   %al,%al
  408dd1:	75 0d                	jne    408de0 <__sprintf_chk@plt+0x6550>
  408dd3:	84 d2                	test   %dl,%dl
  408dd5:	b8 01 00 00 00       	mov    $0x1,%eax
  408dda:	74 04                	je     408de0 <__sprintf_chk@plt+0x6550>
  408ddc:	f3 c3                	repz retq 
  408dde:	66 90                	xchg   %ax,%ax
  408de0:	48 8b 4e 78          	mov    0x78(%rsi),%rcx
  408de4:	48 39 4f 78          	cmp    %rcx,0x78(%rdi)
  408de8:	48 8b 87 80 00 00 00 	mov    0x80(%rdi),%rax
  408def:	48 8b 96 80 00 00 00 	mov    0x80(%rsi),%rdx
  408df6:	7f 20                	jg     408e18 <__sprintf_chk@plt+0x6588>
  408df8:	7c 26                	jl     408e20 <__sprintf_chk@plt+0x6590>
  408dfa:	29 c2                	sub    %eax,%edx
  408dfc:	75 28                	jne    408e26 <__sprintf_chk@plt+0x6596>
  408dfe:	48 8b 36             	mov    (%rsi),%rsi
  408e01:	48 8b 3f             	mov    (%rdi),%rdi
  408e04:	e9 17 c2 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408e09:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408e10:	84 d2                	test   %dl,%dl
  408e12:	75 bb                	jne    408dcf <__sprintf_chk@plt+0x653f>
  408e14:	0f 1f 40 00          	nopl   0x0(%rax)
  408e18:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408e1d:	c3                   	retq   
  408e1e:	66 90                	xchg   %ax,%ax
  408e20:	b8 01 00 00 00       	mov    $0x1,%eax
  408e25:	c3                   	retq   
  408e26:	89 d0                	mov    %edx,%eax
  408e28:	c3                   	retq   
  408e29:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408e30:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408e36:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408e3d:	49 89 f1             	mov    %rsi,%r9
  408e40:	83 f8 09             	cmp    $0x9,%eax
  408e43:	0f 94 c1             	sete   %cl
  408e46:	83 f8 03             	cmp    $0x3,%eax
  408e49:	0f 94 c0             	sete   %al
  408e4c:	41 83 f8 09          	cmp    $0x9,%r8d
  408e50:	0f 94 c2             	sete   %dl
  408e53:	41 83 f8 03          	cmp    $0x3,%r8d
  408e57:	40 0f 94 c6          	sete   %sil
  408e5b:	09 f2                	or     %esi,%edx
  408e5d:	08 c8                	or     %cl,%al
  408e5f:	75 47                	jne    408ea8 <__sprintf_chk@plt+0x6618>
  408e61:	84 c0                	test   %al,%al
  408e63:	75 13                	jne    408e78 <__sprintf_chk@plt+0x65e8>
  408e65:	84 d2                	test   %dl,%dl
  408e67:	b8 01 00 00 00       	mov    $0x1,%eax
  408e6c:	74 0a                	je     408e78 <__sprintf_chk@plt+0x65e8>
  408e6e:	66 90                	xchg   %ax,%ax
  408e70:	f3 c3                	repz retq 
  408e72:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408e78:	48 8b 4f 78          	mov    0x78(%rdi),%rcx
  408e7c:	49 39 49 78          	cmp    %rcx,0x78(%r9)
  408e80:	49 8b 81 80 00 00 00 	mov    0x80(%r9),%rax
  408e87:	48 8b 97 80 00 00 00 	mov    0x80(%rdi),%rdx
  408e8e:	7f 20                	jg     408eb0 <__sprintf_chk@plt+0x6620>
  408e90:	7c 2e                	jl     408ec0 <__sprintf_chk@plt+0x6630>
  408e92:	29 c2                	sub    %eax,%edx
  408e94:	75 30                	jne    408ec6 <__sprintf_chk@plt+0x6636>
  408e96:	48 8b 37             	mov    (%rdi),%rsi
  408e99:	49 8b 39             	mov    (%r9),%rdi
  408e9c:	e9 7f c1 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408ea1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408ea8:	84 d2                	test   %dl,%dl
  408eaa:	75 b5                	jne    408e61 <__sprintf_chk@plt+0x65d1>
  408eac:	0f 1f 40 00          	nopl   0x0(%rax)
  408eb0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408eb5:	c3                   	retq   
  408eb6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  408ebd:	00 00 00 
  408ec0:	b8 01 00 00 00       	mov    $0x1,%eax
  408ec5:	c3                   	retq   
  408ec6:	89 d0                	mov    %edx,%eax
  408ec8:	c3                   	retq   
  408ec9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408ed0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408ed6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408edd:	83 f8 09             	cmp    $0x9,%eax
  408ee0:	0f 94 c1             	sete   %cl
  408ee3:	83 f8 03             	cmp    $0x3,%eax
  408ee6:	0f 94 c0             	sete   %al
  408ee9:	41 83 f8 09          	cmp    $0x9,%r8d
  408eed:	0f 94 c2             	sete   %dl
  408ef0:	41 83 f8 03          	cmp    $0x3,%r8d
  408ef4:	41 0f 94 c0          	sete   %r8b
  408ef8:	44 09 c2             	or     %r8d,%edx
  408efb:	08 c8                	or     %cl,%al
  408efd:	75 39                	jne    408f38 <__sprintf_chk@plt+0x66a8>
  408eff:	84 c0                	test   %al,%al
  408f01:	75 0d                	jne    408f10 <__sprintf_chk@plt+0x6680>
  408f03:	84 d2                	test   %dl,%dl
  408f05:	b8 01 00 00 00       	mov    $0x1,%eax
  408f0a:	74 04                	je     408f10 <__sprintf_chk@plt+0x6680>
  408f0c:	f3 c3                	repz retq 
  408f0e:	66 90                	xchg   %ax,%ax
  408f10:	48 8b 4e 58          	mov    0x58(%rsi),%rcx
  408f14:	48 39 4f 58          	cmp    %rcx,0x58(%rdi)
  408f18:	48 8b 47 60          	mov    0x60(%rdi),%rax
  408f1c:	48 8b 56 60          	mov    0x60(%rsi),%rdx
  408f20:	7f 1e                	jg     408f40 <__sprintf_chk@plt+0x66b0>
  408f22:	7c 2c                	jl     408f50 <__sprintf_chk@plt+0x66c0>
  408f24:	29 c2                	sub    %eax,%edx
  408f26:	75 2e                	jne    408f56 <__sprintf_chk@plt+0x66c6>
  408f28:	48 8b 36             	mov    (%rsi),%rsi
  408f2b:	48 8b 3f             	mov    (%rdi),%rdi
  408f2e:	e9 ed c0 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408f33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  408f38:	84 d2                	test   %dl,%dl
  408f3a:	75 c3                	jne    408eff <__sprintf_chk@plt+0x666f>
  408f3c:	0f 1f 40 00          	nopl   0x0(%rax)
  408f40:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408f45:	c3                   	retq   
  408f46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  408f4d:	00 00 00 
  408f50:	b8 01 00 00 00       	mov    $0x1,%eax
  408f55:	c3                   	retq   
  408f56:	89 d0                	mov    %edx,%eax
  408f58:	c3                   	retq   
  408f59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408f60:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  408f66:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  408f6d:	49 89 f1             	mov    %rsi,%r9
  408f70:	83 f8 09             	cmp    $0x9,%eax
  408f73:	0f 94 c1             	sete   %cl
  408f76:	83 f8 03             	cmp    $0x3,%eax
  408f79:	0f 94 c0             	sete   %al
  408f7c:	41 83 f8 09          	cmp    $0x9,%r8d
  408f80:	0f 94 c2             	sete   %dl
  408f83:	41 83 f8 03          	cmp    $0x3,%r8d
  408f87:	40 0f 94 c6          	sete   %sil
  408f8b:	09 f2                	or     %esi,%edx
  408f8d:	08 c8                	or     %cl,%al
  408f8f:	75 3f                	jne    408fd0 <__sprintf_chk@plt+0x6740>
  408f91:	84 c0                	test   %al,%al
  408f93:	75 13                	jne    408fa8 <__sprintf_chk@plt+0x6718>
  408f95:	84 d2                	test   %dl,%dl
  408f97:	b8 01 00 00 00       	mov    $0x1,%eax
  408f9c:	74 0a                	je     408fa8 <__sprintf_chk@plt+0x6718>
  408f9e:	66 90                	xchg   %ax,%ax
  408fa0:	f3 c3                	repz retq 
  408fa2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  408fa8:	48 8b 4f 58          	mov    0x58(%rdi),%rcx
  408fac:	49 39 49 58          	cmp    %rcx,0x58(%r9)
  408fb0:	49 8b 41 60          	mov    0x60(%r9),%rax
  408fb4:	48 8b 57 60          	mov    0x60(%rdi),%rdx
  408fb8:	7f 1e                	jg     408fd8 <__sprintf_chk@plt+0x6748>
  408fba:	7c 24                	jl     408fe0 <__sprintf_chk@plt+0x6750>
  408fbc:	29 c2                	sub    %eax,%edx
  408fbe:	75 26                	jne    408fe6 <__sprintf_chk@plt+0x6756>
  408fc0:	48 8b 37             	mov    (%rdi),%rsi
  408fc3:	49 8b 39             	mov    (%r9),%rdi
  408fc6:	e9 55 c0 ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  408fcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  408fd0:	84 d2                	test   %dl,%dl
  408fd2:	75 bd                	jne    408f91 <__sprintf_chk@plt+0x6701>
  408fd4:	0f 1f 40 00          	nopl   0x0(%rax)
  408fd8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  408fdd:	c3                   	retq   
  408fde:	66 90                	xchg   %ax,%ax
  408fe0:	b8 01 00 00 00       	mov    $0x1,%eax
  408fe5:	c3                   	retq   
  408fe6:	89 d0                	mov    %edx,%eax
  408fe8:	c3                   	retq   
  408fe9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  408ff0:	48 8b 4e 68          	mov    0x68(%rsi),%rcx
  408ff4:	48 39 4f 68          	cmp    %rcx,0x68(%rdi)
  408ff8:	48 8b 57 70          	mov    0x70(%rdi),%rdx
  408ffc:	48 8b 46 70          	mov    0x70(%rsi),%rax
  409000:	7f 16                	jg     409018 <__sprintf_chk@plt+0x6788>
  409002:	7c 1c                	jl     409020 <__sprintf_chk@plt+0x6790>
  409004:	29 d0                	sub    %edx,%eax
  409006:	75 1d                	jne    409025 <__sprintf_chk@plt+0x6795>
  409008:	48 8b 36             	mov    (%rsi),%rsi
  40900b:	48 8b 3f             	mov    (%rdi),%rdi
  40900e:	e9 3d 95 ff ff       	jmpq   402550 <strcmp@plt>
  409013:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409018:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40901d:	c3                   	retq   
  40901e:	66 90                	xchg   %ax,%ax
  409020:	b8 01 00 00 00       	mov    $0x1,%eax
  409025:	c3                   	retq   
  409026:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40902d:	00 00 00 
  409030:	48 89 f0             	mov    %rsi,%rax
  409033:	48 8b 96 80 00 00 00 	mov    0x80(%rsi),%rdx
  40903a:	48 8b 77 78          	mov    0x78(%rdi),%rsi
  40903e:	48 39 70 78          	cmp    %rsi,0x78(%rax)
  409042:	48 8b 8f 80 00 00 00 	mov    0x80(%rdi),%rcx
  409049:	7f 15                	jg     409060 <__sprintf_chk@plt+0x67d0>
  40904b:	7c 23                	jl     409070 <__sprintf_chk@plt+0x67e0>
  40904d:	29 d1                	sub    %edx,%ecx
  40904f:	75 25                	jne    409076 <__sprintf_chk@plt+0x67e6>
  409051:	48 8b 37             	mov    (%rdi),%rsi
  409054:	48 8b 38             	mov    (%rax),%rdi
  409057:	e9 f4 94 ff ff       	jmpq   402550 <strcmp@plt>
  40905c:	0f 1f 40 00          	nopl   0x0(%rax)
  409060:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409065:	c3                   	retq   
  409066:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40906d:	00 00 00 
  409070:	b8 01 00 00 00       	mov    $0x1,%eax
  409075:	c3                   	retq   
  409076:	89 c8                	mov    %ecx,%eax
  409078:	c3                   	retq   
  409079:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409080:	48 8b 4e 78          	mov    0x78(%rsi),%rcx
  409084:	48 39 4f 78          	cmp    %rcx,0x78(%rdi)
  409088:	48 8b 97 80 00 00 00 	mov    0x80(%rdi),%rdx
  40908f:	48 8b 86 80 00 00 00 	mov    0x80(%rsi),%rax
  409096:	7f 18                	jg     4090b0 <__sprintf_chk@plt+0x6820>
  409098:	7c 26                	jl     4090c0 <__sprintf_chk@plt+0x6830>
  40909a:	29 d0                	sub    %edx,%eax
  40909c:	75 27                	jne    4090c5 <__sprintf_chk@plt+0x6835>
  40909e:	48 8b 36             	mov    (%rsi),%rsi
  4090a1:	48 8b 3f             	mov    (%rdi),%rdi
  4090a4:	e9 a7 94 ff ff       	jmpq   402550 <strcmp@plt>
  4090a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4090b0:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4090b5:	c3                   	retq   
  4090b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4090bd:	00 00 00 
  4090c0:	b8 01 00 00 00       	mov    $0x1,%eax
  4090c5:	c3                   	retq   
  4090c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4090cd:	00 00 00 
  4090d0:	48 89 f0             	mov    %rsi,%rax
  4090d3:	48 8b 56 70          	mov    0x70(%rsi),%rdx
  4090d7:	48 8b 77 68          	mov    0x68(%rdi),%rsi
  4090db:	48 39 70 68          	cmp    %rsi,0x68(%rax)
  4090df:	48 8b 4f 70          	mov    0x70(%rdi),%rcx
  4090e3:	7f 1b                	jg     409100 <__sprintf_chk@plt+0x6870>
  4090e5:	7c 29                	jl     409110 <__sprintf_chk@plt+0x6880>
  4090e7:	29 d1                	sub    %edx,%ecx
  4090e9:	75 2b                	jne    409116 <__sprintf_chk@plt+0x6886>
  4090eb:	48 8b 37             	mov    (%rdi),%rsi
  4090ee:	48 8b 38             	mov    (%rax),%rdi
  4090f1:	e9 5a 94 ff ff       	jmpq   402550 <strcmp@plt>
  4090f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4090fd:	00 00 00 
  409100:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409105:	c3                   	retq   
  409106:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40910d:	00 00 00 
  409110:	b8 01 00 00 00       	mov    $0x1,%eax
  409115:	c3                   	retq   
  409116:	89 c8                	mov    %ecx,%eax
  409118:	c3                   	retq   
  409119:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409120:	48 8b 4e 58          	mov    0x58(%rsi),%rcx
  409124:	48 39 4f 58          	cmp    %rcx,0x58(%rdi)
  409128:	48 8b 57 60          	mov    0x60(%rdi),%rdx
  40912c:	48 8b 46 60          	mov    0x60(%rsi),%rax
  409130:	7f 16                	jg     409148 <__sprintf_chk@plt+0x68b8>
  409132:	7c 1c                	jl     409150 <__sprintf_chk@plt+0x68c0>
  409134:	29 d0                	sub    %edx,%eax
  409136:	75 1d                	jne    409155 <__sprintf_chk@plt+0x68c5>
  409138:	48 8b 36             	mov    (%rsi),%rsi
  40913b:	48 8b 3f             	mov    (%rdi),%rdi
  40913e:	e9 0d 94 ff ff       	jmpq   402550 <strcmp@plt>
  409143:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409148:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40914d:	c3                   	retq   
  40914e:	66 90                	xchg   %ax,%ax
  409150:	b8 01 00 00 00       	mov    $0x1,%eax
  409155:	c3                   	retq   
  409156:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40915d:	00 00 00 
  409160:	48 89 f0             	mov    %rsi,%rax
  409163:	48 8b 56 60          	mov    0x60(%rsi),%rdx
  409167:	48 8b 77 58          	mov    0x58(%rdi),%rsi
  40916b:	48 39 70 58          	cmp    %rsi,0x58(%rax)
  40916f:	48 8b 4f 60          	mov    0x60(%rdi),%rcx
  409173:	7f 1b                	jg     409190 <__sprintf_chk@plt+0x6900>
  409175:	7c 29                	jl     4091a0 <__sprintf_chk@plt+0x6910>
  409177:	29 d1                	sub    %edx,%ecx
  409179:	75 2b                	jne    4091a6 <__sprintf_chk@plt+0x6916>
  40917b:	48 8b 37             	mov    (%rdi),%rsi
  40917e:	48 8b 38             	mov    (%rax),%rdi
  409181:	e9 ca 93 ff ff       	jmpq   402550 <strcmp@plt>
  409186:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40918d:	00 00 00 
  409190:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409195:	c3                   	retq   
  409196:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40919d:	00 00 00 
  4091a0:	b8 01 00 00 00       	mov    $0x1,%eax
  4091a5:	c3                   	retq   
  4091a6:	89 c8                	mov    %ecx,%eax
  4091a8:	c3                   	retq   
  4091a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4091b0:	41 54                	push   %r12
  4091b2:	49 89 f4             	mov    %rsi,%r12
  4091b5:	be 2e 00 00 00       	mov    $0x2e,%esi
  4091ba:	55                   	push   %rbp
  4091bb:	48 89 fd             	mov    %rdi,%rbp
  4091be:	49 8b 3c 24          	mov    (%r12),%rdi
  4091c2:	53                   	push   %rbx
  4091c3:	e8 48 92 ff ff       	callq  402410 <strrchr@plt>
  4091c8:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  4091cc:	be 2e 00 00 00       	mov    $0x2e,%esi
  4091d1:	48 89 c3             	mov    %rax,%rbx
  4091d4:	e8 37 92 ff ff       	callq  402410 <strrchr@plt>
  4091d9:	ba 19 69 41 00       	mov    $0x416919,%edx
  4091de:	48 85 c0             	test   %rax,%rax
  4091e1:	48 0f 44 c2          	cmove  %rdx,%rax
  4091e5:	48 85 db             	test   %rbx,%rbx
  4091e8:	48 0f 45 d3          	cmovne %rbx,%rdx
  4091ec:	48 89 c6             	mov    %rax,%rsi
  4091ef:	48 89 d7             	mov    %rdx,%rdi
  4091f2:	e8 29 be ff ff       	callq  405020 <__sprintf_chk@plt+0x2790>
  4091f7:	85 c0                	test   %eax,%eax
  4091f9:	74 05                	je     409200 <__sprintf_chk@plt+0x6970>
  4091fb:	5b                   	pop    %rbx
  4091fc:	5d                   	pop    %rbp
  4091fd:	41 5c                	pop    %r12
  4091ff:	c3                   	retq   
  409200:	5b                   	pop    %rbx
  409201:	48 8b 75 00          	mov    0x0(%rbp),%rsi
  409205:	49 8b 3c 24          	mov    (%r12),%rdi
  409209:	5d                   	pop    %rbp
  40920a:	41 5c                	pop    %r12
  40920c:	e9 0f be ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  409211:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  409218:	0f 1f 84 00 00 00 00 
  40921f:	00 
  409220:	41 54                	push   %r12
  409222:	49 89 fc             	mov    %rdi,%r12
  409225:	55                   	push   %rbp
  409226:	48 89 f5             	mov    %rsi,%rbp
  409229:	be 2e 00 00 00       	mov    $0x2e,%esi
  40922e:	53                   	push   %rbx
  40922f:	48 8b 3f             	mov    (%rdi),%rdi
  409232:	e8 d9 91 ff ff       	callq  402410 <strrchr@plt>
  409237:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  40923b:	be 2e 00 00 00       	mov    $0x2e,%esi
  409240:	48 89 c3             	mov    %rax,%rbx
  409243:	e8 c8 91 ff ff       	callq  402410 <strrchr@plt>
  409248:	ba 19 69 41 00       	mov    $0x416919,%edx
  40924d:	48 85 c0             	test   %rax,%rax
  409250:	48 0f 44 c2          	cmove  %rdx,%rax
  409254:	48 85 db             	test   %rbx,%rbx
  409257:	48 0f 45 d3          	cmovne %rbx,%rdx
  40925b:	48 89 c6             	mov    %rax,%rsi
  40925e:	48 89 d7             	mov    %rdx,%rdi
  409261:	e8 ba bd ff ff       	callq  405020 <__sprintf_chk@plt+0x2790>
  409266:	85 c0                	test   %eax,%eax
  409268:	74 06                	je     409270 <__sprintf_chk@plt+0x69e0>
  40926a:	5b                   	pop    %rbx
  40926b:	5d                   	pop    %rbp
  40926c:	41 5c                	pop    %r12
  40926e:	c3                   	retq   
  40926f:	90                   	nop
  409270:	5b                   	pop    %rbx
  409271:	48 8b 75 00          	mov    0x0(%rbp),%rsi
  409275:	49 8b 3c 24          	mov    (%r12),%rdi
  409279:	5d                   	pop    %rbp
  40927a:	41 5c                	pop    %r12
  40927c:	e9 9f bd ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  409281:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  409288:	0f 1f 84 00 00 00 00 
  40928f:	00 
  409290:	41 54                	push   %r12
  409292:	55                   	push   %rbp
  409293:	48 89 f5             	mov    %rsi,%rbp
  409296:	53                   	push   %rbx
  409297:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  40929d:	48 89 fb             	mov    %rdi,%rbx
  4092a0:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  4092a7:	83 f8 09             	cmp    $0x9,%eax
  4092aa:	0f 94 c1             	sete   %cl
  4092ad:	83 f8 03             	cmp    $0x3,%eax
  4092b0:	0f 94 c0             	sete   %al
  4092b3:	41 83 f8 09          	cmp    $0x9,%r8d
  4092b7:	0f 94 c2             	sete   %dl
  4092ba:	41 83 f8 03          	cmp    $0x3,%r8d
  4092be:	40 0f 94 c6          	sete   %sil
  4092c2:	09 f2                	or     %esi,%edx
  4092c4:	08 c8                	or     %cl,%al
  4092c6:	75 68                	jne    409330 <__sprintf_chk@plt+0x6aa0>
  4092c8:	84 c0                	test   %al,%al
  4092ca:	75 14                	jne    4092e0 <__sprintf_chk@plt+0x6a50>
  4092cc:	84 d2                	test   %dl,%dl
  4092ce:	b8 01 00 00 00       	mov    $0x1,%eax
  4092d3:	74 0b                	je     4092e0 <__sprintf_chk@plt+0x6a50>
  4092d5:	5b                   	pop    %rbx
  4092d6:	5d                   	pop    %rbp
  4092d7:	41 5c                	pop    %r12
  4092d9:	c3                   	retq   
  4092da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4092e0:	48 8b 3b             	mov    (%rbx),%rdi
  4092e3:	be 2e 00 00 00       	mov    $0x2e,%esi
  4092e8:	e8 23 91 ff ff       	callq  402410 <strrchr@plt>
  4092ed:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  4092f1:	be 2e 00 00 00       	mov    $0x2e,%esi
  4092f6:	49 89 c4             	mov    %rax,%r12
  4092f9:	e8 12 91 ff ff       	callq  402410 <strrchr@plt>
  4092fe:	bf 19 69 41 00       	mov    $0x416919,%edi
  409303:	48 85 c0             	test   %rax,%rax
  409306:	48 0f 44 c7          	cmove  %rdi,%rax
  40930a:	4d 85 e4             	test   %r12,%r12
  40930d:	49 0f 45 fc          	cmovne %r12,%rdi
  409311:	48 89 c6             	mov    %rax,%rsi
  409314:	e8 07 bd ff ff       	callq  405020 <__sprintf_chk@plt+0x2790>
  409319:	85 c0                	test   %eax,%eax
  40931b:	75 b8                	jne    4092d5 <__sprintf_chk@plt+0x6a45>
  40931d:	48 8b 3b             	mov    (%rbx),%rdi
  409320:	48 8b 75 00          	mov    0x0(%rbp),%rsi
  409324:	5b                   	pop    %rbx
  409325:	5d                   	pop    %rbp
  409326:	41 5c                	pop    %r12
  409328:	e9 f3 bc ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  40932d:	0f 1f 00             	nopl   (%rax)
  409330:	84 d2                	test   %dl,%dl
  409332:	75 94                	jne    4092c8 <__sprintf_chk@plt+0x6a38>
  409334:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409339:	eb 9a                	jmp    4092d5 <__sprintf_chk@plt+0x6a45>
  40933b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409340:	41 54                	push   %r12
  409342:	55                   	push   %rbp
  409343:	48 89 fd             	mov    %rdi,%rbp
  409346:	53                   	push   %rbx
  409347:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  40934d:	48 89 f3             	mov    %rsi,%rbx
  409350:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  409357:	83 f8 09             	cmp    $0x9,%eax
  40935a:	0f 94 c1             	sete   %cl
  40935d:	83 f8 03             	cmp    $0x3,%eax
  409360:	0f 94 c0             	sete   %al
  409363:	41 83 f8 09          	cmp    $0x9,%r8d
  409367:	0f 94 c2             	sete   %dl
  40936a:	41 83 f8 03          	cmp    $0x3,%r8d
  40936e:	40 0f 94 c6          	sete   %sil
  409372:	09 f2                	or     %esi,%edx
  409374:	08 c8                	or     %cl,%al
  409376:	75 68                	jne    4093e0 <__sprintf_chk@plt+0x6b50>
  409378:	84 c0                	test   %al,%al
  40937a:	75 14                	jne    409390 <__sprintf_chk@plt+0x6b00>
  40937c:	84 d2                	test   %dl,%dl
  40937e:	b8 01 00 00 00       	mov    $0x1,%eax
  409383:	74 0b                	je     409390 <__sprintf_chk@plt+0x6b00>
  409385:	5b                   	pop    %rbx
  409386:	5d                   	pop    %rbp
  409387:	41 5c                	pop    %r12
  409389:	c3                   	retq   
  40938a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409390:	48 8b 3b             	mov    (%rbx),%rdi
  409393:	be 2e 00 00 00       	mov    $0x2e,%esi
  409398:	e8 73 90 ff ff       	callq  402410 <strrchr@plt>
  40939d:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  4093a1:	be 2e 00 00 00       	mov    $0x2e,%esi
  4093a6:	49 89 c4             	mov    %rax,%r12
  4093a9:	e8 62 90 ff ff       	callq  402410 <strrchr@plt>
  4093ae:	bf 19 69 41 00       	mov    $0x416919,%edi
  4093b3:	48 85 c0             	test   %rax,%rax
  4093b6:	48 0f 44 c7          	cmove  %rdi,%rax
  4093ba:	4d 85 e4             	test   %r12,%r12
  4093bd:	49 0f 45 fc          	cmovne %r12,%rdi
  4093c1:	48 89 c6             	mov    %rax,%rsi
  4093c4:	e8 57 bc ff ff       	callq  405020 <__sprintf_chk@plt+0x2790>
  4093c9:	85 c0                	test   %eax,%eax
  4093cb:	75 b8                	jne    409385 <__sprintf_chk@plt+0x6af5>
  4093cd:	48 8b 3b             	mov    (%rbx),%rdi
  4093d0:	48 8b 75 00          	mov    0x0(%rbp),%rsi
  4093d4:	5b                   	pop    %rbx
  4093d5:	5d                   	pop    %rbp
  4093d6:	41 5c                	pop    %r12
  4093d8:	e9 43 bc ff ff       	jmpq   405020 <__sprintf_chk@plt+0x2790>
  4093dd:	0f 1f 00             	nopl   (%rax)
  4093e0:	84 d2                	test   %dl,%dl
  4093e2:	75 94                	jne    409378 <__sprintf_chk@plt+0x6ae8>
  4093e4:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4093e9:	eb 9a                	jmp    409385 <__sprintf_chk@plt+0x6af5>
  4093eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4093f0:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  4093f6:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  4093fd:	49 89 f1             	mov    %rsi,%r9
  409400:	83 f8 09             	cmp    $0x9,%eax
  409403:	0f 94 c1             	sete   %cl
  409406:	83 f8 03             	cmp    $0x3,%eax
  409409:	0f 94 c0             	sete   %al
  40940c:	41 83 f8 09          	cmp    $0x9,%r8d
  409410:	0f 94 c2             	sete   %dl
  409413:	41 83 f8 03          	cmp    $0x3,%r8d
  409417:	40 0f 94 c6          	sete   %sil
  40941b:	09 f2                	or     %esi,%edx
  40941d:	08 c8                	or     %cl,%al
  40941f:	75 47                	jne    409468 <__sprintf_chk@plt+0x6bd8>
  409421:	84 c0                	test   %al,%al
  409423:	75 13                	jne    409438 <__sprintf_chk@plt+0x6ba8>
  409425:	84 d2                	test   %dl,%dl
  409427:	b8 01 00 00 00       	mov    $0x1,%eax
  40942c:	74 0a                	je     409438 <__sprintf_chk@plt+0x6ba8>
  40942e:	66 90                	xchg   %ax,%ax
  409430:	f3 c3                	repz retq 
  409432:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409438:	48 8b 4f 78          	mov    0x78(%rdi),%rcx
  40943c:	49 39 49 78          	cmp    %rcx,0x78(%r9)
  409440:	49 8b 81 80 00 00 00 	mov    0x80(%r9),%rax
  409447:	48 8b 97 80 00 00 00 	mov    0x80(%rdi),%rdx
  40944e:	7f 20                	jg     409470 <__sprintf_chk@plt+0x6be0>
  409450:	7c 2e                	jl     409480 <__sprintf_chk@plt+0x6bf0>
  409452:	29 c2                	sub    %eax,%edx
  409454:	75 30                	jne    409486 <__sprintf_chk@plt+0x6bf6>
  409456:	48 8b 37             	mov    (%rdi),%rsi
  409459:	49 8b 39             	mov    (%r9),%rdi
  40945c:	e9 ef 90 ff ff       	jmpq   402550 <strcmp@plt>
  409461:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409468:	84 d2                	test   %dl,%dl
  40946a:	75 b5                	jne    409421 <__sprintf_chk@plt+0x6b91>
  40946c:	0f 1f 40 00          	nopl   0x0(%rax)
  409470:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409475:	c3                   	retq   
  409476:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40947d:	00 00 00 
  409480:	b8 01 00 00 00       	mov    $0x1,%eax
  409485:	c3                   	retq   
  409486:	89 d0                	mov    %edx,%eax
  409488:	c3                   	retq   
  409489:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409490:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  409496:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40949d:	83 f8 09             	cmp    $0x9,%eax
  4094a0:	0f 94 c1             	sete   %cl
  4094a3:	83 f8 03             	cmp    $0x3,%eax
  4094a6:	0f 94 c0             	sete   %al
  4094a9:	41 83 f8 09          	cmp    $0x9,%r8d
  4094ad:	0f 94 c2             	sete   %dl
  4094b0:	41 83 f8 03          	cmp    $0x3,%r8d
  4094b4:	41 0f 94 c0          	sete   %r8b
  4094b8:	44 09 c2             	or     %r8d,%edx
  4094bb:	08 c8                	or     %cl,%al
  4094bd:	75 39                	jne    4094f8 <__sprintf_chk@plt+0x6c68>
  4094bf:	84 c0                	test   %al,%al
  4094c1:	75 0d                	jne    4094d0 <__sprintf_chk@plt+0x6c40>
  4094c3:	84 d2                	test   %dl,%dl
  4094c5:	b8 01 00 00 00       	mov    $0x1,%eax
  4094ca:	74 04                	je     4094d0 <__sprintf_chk@plt+0x6c40>
  4094cc:	f3 c3                	repz retq 
  4094ce:	66 90                	xchg   %ax,%ax
  4094d0:	48 8b 4e 58          	mov    0x58(%rsi),%rcx
  4094d4:	48 39 4f 58          	cmp    %rcx,0x58(%rdi)
  4094d8:	48 8b 47 60          	mov    0x60(%rdi),%rax
  4094dc:	48 8b 56 60          	mov    0x60(%rsi),%rdx
  4094e0:	7f 1e                	jg     409500 <__sprintf_chk@plt+0x6c70>
  4094e2:	7c 2c                	jl     409510 <__sprintf_chk@plt+0x6c80>
  4094e4:	29 c2                	sub    %eax,%edx
  4094e6:	75 2e                	jne    409516 <__sprintf_chk@plt+0x6c86>
  4094e8:	48 8b 36             	mov    (%rsi),%rsi
  4094eb:	48 8b 3f             	mov    (%rdi),%rdi
  4094ee:	e9 5d 90 ff ff       	jmpq   402550 <strcmp@plt>
  4094f3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4094f8:	84 d2                	test   %dl,%dl
  4094fa:	75 c3                	jne    4094bf <__sprintf_chk@plt+0x6c2f>
  4094fc:	0f 1f 40 00          	nopl   0x0(%rax)
  409500:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409505:	c3                   	retq   
  409506:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40950d:	00 00 00 
  409510:	b8 01 00 00 00       	mov    $0x1,%eax
  409515:	c3                   	retq   
  409516:	89 d0                	mov    %edx,%eax
  409518:	c3                   	retq   
  409519:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409520:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  409526:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  40952d:	83 f8 09             	cmp    $0x9,%eax
  409530:	0f 94 c1             	sete   %cl
  409533:	83 f8 03             	cmp    $0x3,%eax
  409536:	0f 94 c0             	sete   %al
  409539:	41 83 f8 09          	cmp    $0x9,%r8d
  40953d:	0f 94 c2             	sete   %dl
  409540:	41 83 f8 03          	cmp    $0x3,%r8d
  409544:	41 0f 94 c0          	sete   %r8b
  409548:	44 09 c2             	or     %r8d,%edx
  40954b:	08 c8                	or     %cl,%al
  40954d:	75 39                	jne    409588 <__sprintf_chk@plt+0x6cf8>
  40954f:	84 c0                	test   %al,%al
  409551:	75 0d                	jne    409560 <__sprintf_chk@plt+0x6cd0>
  409553:	84 d2                	test   %dl,%dl
  409555:	b8 01 00 00 00       	mov    $0x1,%eax
  40955a:	74 04                	je     409560 <__sprintf_chk@plt+0x6cd0>
  40955c:	f3 c3                	repz retq 
  40955e:	66 90                	xchg   %ax,%ax
  409560:	48 8b 4e 68          	mov    0x68(%rsi),%rcx
  409564:	48 39 4f 68          	cmp    %rcx,0x68(%rdi)
  409568:	48 8b 47 70          	mov    0x70(%rdi),%rax
  40956c:	48 8b 56 70          	mov    0x70(%rsi),%rdx
  409570:	7f 1e                	jg     409590 <__sprintf_chk@plt+0x6d00>
  409572:	7c 2c                	jl     4095a0 <__sprintf_chk@plt+0x6d10>
  409574:	29 c2                	sub    %eax,%edx
  409576:	75 2e                	jne    4095a6 <__sprintf_chk@plt+0x6d16>
  409578:	48 8b 36             	mov    (%rsi),%rsi
  40957b:	48 8b 3f             	mov    (%rdi),%rdi
  40957e:	e9 cd 8f ff ff       	jmpq   402550 <strcmp@plt>
  409583:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409588:	84 d2                	test   %dl,%dl
  40958a:	75 c3                	jne    40954f <__sprintf_chk@plt+0x6cbf>
  40958c:	0f 1f 40 00          	nopl   0x0(%rax)
  409590:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409595:	c3                   	retq   
  409596:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40959d:	00 00 00 
  4095a0:	b8 01 00 00 00       	mov    $0x1,%eax
  4095a5:	c3                   	retq   
  4095a6:	89 d0                	mov    %edx,%eax
  4095a8:	c3                   	retq   
  4095a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4095b0:	41 54                	push   %r12
  4095b2:	49 89 fc             	mov    %rdi,%r12
  4095b5:	55                   	push   %rbp
  4095b6:	53                   	push   %rbx
  4095b7:	48 8b 2e             	mov    (%rsi),%rbp
  4095ba:	be 2e 00 00 00       	mov    $0x2e,%esi
  4095bf:	48 89 ef             	mov    %rbp,%rdi
  4095c2:	e8 49 8e ff ff       	callq  402410 <strrchr@plt>
  4095c7:	4d 8b 24 24          	mov    (%r12),%r12
  4095cb:	be 2e 00 00 00       	mov    $0x2e,%esi
  4095d0:	48 89 c3             	mov    %rax,%rbx
  4095d3:	4c 89 e7             	mov    %r12,%rdi
  4095d6:	e8 35 8e ff ff       	callq  402410 <strrchr@plt>
  4095db:	ba 19 69 41 00       	mov    $0x416919,%edx
  4095e0:	48 85 c0             	test   %rax,%rax
  4095e3:	48 0f 44 c2          	cmove  %rdx,%rax
  4095e7:	48 85 db             	test   %rbx,%rbx
  4095ea:	48 0f 45 d3          	cmovne %rbx,%rdx
  4095ee:	48 89 c6             	mov    %rax,%rsi
  4095f1:	48 89 d7             	mov    %rdx,%rdi
  4095f4:	e8 57 8f ff ff       	callq  402550 <strcmp@plt>
  4095f9:	85 c0                	test   %eax,%eax
  4095fb:	74 0b                	je     409608 <__sprintf_chk@plt+0x6d78>
  4095fd:	5b                   	pop    %rbx
  4095fe:	5d                   	pop    %rbp
  4095ff:	41 5c                	pop    %r12
  409601:	c3                   	retq   
  409602:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409608:	5b                   	pop    %rbx
  409609:	48 89 ef             	mov    %rbp,%rdi
  40960c:	4c 89 e6             	mov    %r12,%rsi
  40960f:	5d                   	pop    %rbp
  409610:	41 5c                	pop    %r12
  409612:	e9 39 8f ff ff       	jmpq   402550 <strcmp@plt>
  409617:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40961e:	00 00 
  409620:	41 54                	push   %r12
  409622:	49 89 f4             	mov    %rsi,%r12
  409625:	be 2e 00 00 00       	mov    $0x2e,%esi
  40962a:	55                   	push   %rbp
  40962b:	53                   	push   %rbx
  40962c:	48 8b 2f             	mov    (%rdi),%rbp
  40962f:	48 89 ef             	mov    %rbp,%rdi
  409632:	e8 d9 8d ff ff       	callq  402410 <strrchr@plt>
  409637:	4d 8b 24 24          	mov    (%r12),%r12
  40963b:	be 2e 00 00 00       	mov    $0x2e,%esi
  409640:	48 89 c3             	mov    %rax,%rbx
  409643:	4c 89 e7             	mov    %r12,%rdi
  409646:	e8 c5 8d ff ff       	callq  402410 <strrchr@plt>
  40964b:	ba 19 69 41 00       	mov    $0x416919,%edx
  409650:	48 85 c0             	test   %rax,%rax
  409653:	48 0f 44 c2          	cmove  %rdx,%rax
  409657:	48 85 db             	test   %rbx,%rbx
  40965a:	48 0f 45 d3          	cmovne %rbx,%rdx
  40965e:	48 89 c6             	mov    %rax,%rsi
  409661:	48 89 d7             	mov    %rdx,%rdi
  409664:	e8 e7 8e ff ff       	callq  402550 <strcmp@plt>
  409669:	85 c0                	test   %eax,%eax
  40966b:	74 0b                	je     409678 <__sprintf_chk@plt+0x6de8>
  40966d:	5b                   	pop    %rbx
  40966e:	5d                   	pop    %rbp
  40966f:	41 5c                	pop    %r12
  409671:	c3                   	retq   
  409672:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409678:	5b                   	pop    %rbx
  409679:	48 89 ef             	mov    %rbp,%rdi
  40967c:	4c 89 e6             	mov    %r12,%rsi
  40967f:	5d                   	pop    %rbp
  409680:	41 5c                	pop    %r12
  409682:	e9 c9 8e ff ff       	jmpq   402550 <strcmp@plt>
  409687:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40968e:	00 00 
  409690:	41 54                	push   %r12
  409692:	55                   	push   %rbp
  409693:	53                   	push   %rbx
  409694:	8b 87 a0 00 00 00    	mov    0xa0(%rdi),%eax
  40969a:	48 89 fb             	mov    %rdi,%rbx
  40969d:	44 8b 86 a0 00 00 00 	mov    0xa0(%rsi),%r8d
  4096a4:	83 f8 09             	cmp    $0x9,%eax
  4096a7:	0f 94 c1             	sete   %cl
  4096aa:	83 f8 03             	cmp    $0x3,%eax
  4096ad:	0f 94 c0             	sete   %al
  4096b0:	41 83 f8 09          	cmp    $0x9,%r8d
  4096b4:	0f 94 c2             	sete   %dl
  4096b7:	41 83 f8 03          	cmp    $0x3,%r8d
  4096bb:	40 0f 94 c7          	sete   %dil
  4096bf:	09 fa                	or     %edi,%edx
  4096c1:	08 c8                	or     %cl,%al
  4096c3:	75 73                	jne    409738 <__sprintf_chk@plt+0x6ea8>
  4096c5:	84 c0                	test   %al,%al
  4096c7:	75 17                	jne    4096e0 <__sprintf_chk@plt+0x6e50>
  4096c9:	84 d2                	test   %dl,%dl
  4096cb:	b8 01 00 00 00       	mov    $0x1,%eax
  4096d0:	74 0e                	je     4096e0 <__sprintf_chk@plt+0x6e50>
  4096d2:	5b                   	pop    %rbx
  4096d3:	5d                   	pop    %rbp
  4096d4:	41 5c                	pop    %r12
  4096d6:	c3                   	retq   
  4096d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4096de:	00 00 
  4096e0:	4c 8b 26             	mov    (%rsi),%r12
  4096e3:	be 2e 00 00 00       	mov    $0x2e,%esi
  4096e8:	4c 89 e7             	mov    %r12,%rdi
  4096eb:	e8 20 8d ff ff       	callq  402410 <strrchr@plt>
  4096f0:	48 8b 1b             	mov    (%rbx),%rbx
  4096f3:	be 2e 00 00 00       	mov    $0x2e,%esi
  4096f8:	48 89 c5             	mov    %rax,%rbp
  4096fb:	48 89 df             	mov    %rbx,%rdi
  4096fe:	e8 0d 8d ff ff       	callq  402410 <strrchr@plt>
  409703:	ba 19 69 41 00       	mov    $0x416919,%edx
  409708:	48 85 c0             	test   %rax,%rax
  40970b:	48 0f 44 c2          	cmove  %rdx,%rax
  40970f:	48 85 ed             	test   %rbp,%rbp
  409712:	48 0f 45 d5          	cmovne %rbp,%rdx
  409716:	48 89 c6             	mov    %rax,%rsi
  409719:	48 89 d7             	mov    %rdx,%rdi
  40971c:	e8 2f 8e ff ff       	callq  402550 <strcmp@plt>
  409721:	85 c0                	test   %eax,%eax
  409723:	75 ad                	jne    4096d2 <__sprintf_chk@plt+0x6e42>
  409725:	48 89 de             	mov    %rbx,%rsi
  409728:	4c 89 e7             	mov    %r12,%rdi
  40972b:	5b                   	pop    %rbx
  40972c:	5d                   	pop    %rbp
  40972d:	41 5c                	pop    %r12
  40972f:	e9 1c 8e ff ff       	jmpq   402550 <strcmp@plt>
  409734:	0f 1f 40 00          	nopl   0x0(%rax)
  409738:	84 d2                	test   %dl,%dl
  40973a:	75 89                	jne    4096c5 <__sprintf_chk@plt+0x6e35>
  40973c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409741:	eb 8f                	jmp    4096d2 <__sprintf_chk@plt+0x6e42>
  409743:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40974a:	84 00 00 00 00 00 
  409750:	55                   	push   %rbp
  409751:	ba 05 00 00 00       	mov    $0x5,%edx
  409756:	53                   	push   %rbx
  409757:	89 fb                	mov    %edi,%ebx
  409759:	48 83 ec 08          	sub    $0x8,%rsp
  40975d:	85 ff                	test   %edi,%edi
  40975f:	48 8b 2d 9a 1a 21 00 	mov    0x211a9a(%rip),%rbp        # 61b200 <stderr@@GLIBC_2.2.5+0xbb0>
  409766:	74 2c                	je     409794 <__sprintf_chk@plt+0x6f04>
  409768:	be 60 3d 41 00       	mov    $0x413d60,%esi
  40976d:	31 ff                	xor    %edi,%edi
  40976f:	e8 ec 8b ff ff       	callq  402360 <dcgettext@plt>
  409774:	48 8b 3d d5 0e 21 00 	mov    0x210ed5(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  40977b:	48 89 c2             	mov    %rax,%rdx
  40977e:	48 89 e9             	mov    %rbp,%rcx
  409781:	be 01 00 00 00       	mov    $0x1,%esi
  409786:	31 c0                	xor    %eax,%eax
  409788:	e8 83 90 ff ff       	callq  402810 <__fprintf_chk@plt>
  40978d:	89 df                	mov    %ebx,%edi
  40978f:	e8 5c 90 ff ff       	callq  4027f0 <exit@plt>
  409794:	31 ff                	xor    %edi,%edi
  409796:	be 88 3d 41 00       	mov    $0x413d88,%esi
  40979b:	e8 c0 8b ff ff       	callq  402360 <dcgettext@plt>
  4097a0:	48 89 ea             	mov    %rbp,%rdx
  4097a3:	48 89 c6             	mov    %rax,%rsi
  4097a6:	bf 01 00 00 00       	mov    $0x1,%edi
  4097ab:	31 c0                	xor    %eax,%eax
  4097ad:	e8 7e 8f ff ff       	callq  402730 <__printf_chk@plt>
  4097b2:	48 8b 2d 57 0e 21 00 	mov    0x210e57(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4097b9:	ba 05 00 00 00       	mov    $0x5,%edx
  4097be:	31 ff                	xor    %edi,%edi
  4097c0:	be b0 3d 41 00       	mov    $0x413db0,%esi
  4097c5:	e8 96 8b ff ff       	callq  402360 <dcgettext@plt>
  4097ca:	48 89 ee             	mov    %rbp,%rsi
  4097cd:	48 89 c7             	mov    %rax,%rdi
  4097d0:	e8 4b 8d ff ff       	callq  402520 <fputs_unlocked@plt>
  4097d5:	48 8b 2d 34 0e 21 00 	mov    0x210e34(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4097dc:	ba 05 00 00 00       	mov    $0x5,%edx
  4097e1:	31 ff                	xor    %edi,%edi
  4097e3:	be 40 3e 41 00       	mov    $0x413e40,%esi
  4097e8:	e8 73 8b ff ff       	callq  402360 <dcgettext@plt>
  4097ed:	48 89 ee             	mov    %rbp,%rsi
  4097f0:	48 89 c7             	mov    %rax,%rdi
  4097f3:	e8 28 8d ff ff       	callq  402520 <fputs_unlocked@plt>
  4097f8:	48 8b 2d 11 0e 21 00 	mov    0x210e11(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4097ff:	ba 05 00 00 00       	mov    $0x5,%edx
  409804:	31 ff                	xor    %edi,%edi
  409806:	be 90 3e 41 00       	mov    $0x413e90,%esi
  40980b:	e8 50 8b ff ff       	callq  402360 <dcgettext@plt>
  409810:	48 89 ee             	mov    %rbp,%rsi
  409813:	48 89 c7             	mov    %rax,%rdi
  409816:	e8 05 8d ff ff       	callq  402520 <fputs_unlocked@plt>
  40981b:	48 8b 2d ee 0d 21 00 	mov    0x210dee(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409822:	ba 05 00 00 00       	mov    $0x5,%edx
  409827:	31 ff                	xor    %edi,%edi
  409829:	be a0 3f 41 00       	mov    $0x413fa0,%esi
  40982e:	e8 2d 8b ff ff       	callq  402360 <dcgettext@plt>
  409833:	48 89 ee             	mov    %rbp,%rsi
  409836:	48 89 c7             	mov    %rax,%rdi
  409839:	e8 e2 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  40983e:	48 8b 2d cb 0d 21 00 	mov    0x210dcb(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409845:	ba 05 00 00 00       	mov    $0x5,%edx
  40984a:	31 ff                	xor    %edi,%edi
  40984c:	be e8 41 41 00       	mov    $0x4141e8,%esi
  409851:	e8 0a 8b ff ff       	callq  402360 <dcgettext@plt>
  409856:	48 89 ee             	mov    %rbp,%rsi
  409859:	48 89 c7             	mov    %rax,%rdi
  40985c:	e8 bf 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  409861:	48 8b 2d a8 0d 21 00 	mov    0x210da8(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409868:	ba 05 00 00 00       	mov    $0x5,%edx
  40986d:	31 ff                	xor    %edi,%edi
  40986f:	be 98 43 41 00       	mov    $0x414398,%esi
  409874:	e8 e7 8a ff ff       	callq  402360 <dcgettext@plt>
  409879:	48 89 ee             	mov    %rbp,%rsi
  40987c:	48 89 c7             	mov    %rax,%rdi
  40987f:	e8 9c 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  409884:	48 8b 2d 85 0d 21 00 	mov    0x210d85(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  40988b:	ba 05 00 00 00       	mov    $0x5,%edx
  409890:	31 ff                	xor    %edi,%edi
  409892:	be 40 45 41 00       	mov    $0x414540,%esi
  409897:	e8 c4 8a ff ff       	callq  402360 <dcgettext@plt>
  40989c:	48 89 ee             	mov    %rbp,%rsi
  40989f:	48 89 c7             	mov    %rax,%rdi
  4098a2:	e8 79 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  4098a7:	48 8b 2d 62 0d 21 00 	mov    0x210d62(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4098ae:	ba 05 00 00 00       	mov    $0x5,%edx
  4098b3:	31 ff                	xor    %edi,%edi
  4098b5:	be 80 45 41 00       	mov    $0x414580,%esi
  4098ba:	e8 a1 8a ff ff       	callq  402360 <dcgettext@plt>
  4098bf:	48 89 ee             	mov    %rbp,%rsi
  4098c2:	48 89 c7             	mov    %rax,%rdi
  4098c5:	e8 56 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  4098ca:	48 8b 2d 3f 0d 21 00 	mov    0x210d3f(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4098d1:	ba 05 00 00 00       	mov    $0x5,%edx
  4098d6:	31 ff                	xor    %edi,%edi
  4098d8:	be 70 46 41 00       	mov    $0x414670,%esi
  4098dd:	e8 7e 8a ff ff       	callq  402360 <dcgettext@plt>
  4098e2:	48 89 ee             	mov    %rbp,%rsi
  4098e5:	48 89 c7             	mov    %rax,%rdi
  4098e8:	e8 33 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  4098ed:	48 8b 2d 1c 0d 21 00 	mov    0x210d1c(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4098f4:	ba 05 00 00 00       	mov    $0x5,%edx
  4098f9:	31 ff                	xor    %edi,%edi
  4098fb:	be 80 47 41 00       	mov    $0x414780,%esi
  409900:	e8 5b 8a ff ff       	callq  402360 <dcgettext@plt>
  409905:	48 89 ee             	mov    %rbp,%rsi
  409908:	48 89 c7             	mov    %rax,%rdi
  40990b:	e8 10 8c ff ff       	callq  402520 <fputs_unlocked@plt>
  409910:	48 8b 2d f9 0c 21 00 	mov    0x210cf9(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409917:	ba 05 00 00 00       	mov    $0x5,%edx
  40991c:	31 ff                	xor    %edi,%edi
  40991e:	be 28 49 41 00       	mov    $0x414928,%esi
  409923:	e8 38 8a ff ff       	callq  402360 <dcgettext@plt>
  409928:	48 89 ee             	mov    %rbp,%rsi
  40992b:	48 89 c7             	mov    %rax,%rdi
  40992e:	e8 ed 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  409933:	48 8b 2d d6 0c 21 00 	mov    0x210cd6(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  40993a:	ba 05 00 00 00       	mov    $0x5,%edx
  40993f:	31 ff                	xor    %edi,%edi
  409941:	be c0 4a 41 00       	mov    $0x414ac0,%esi
  409946:	e8 15 8a ff ff       	callq  402360 <dcgettext@plt>
  40994b:	48 89 ee             	mov    %rbp,%rsi
  40994e:	48 89 c7             	mov    %rax,%rdi
  409951:	e8 ca 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  409956:	48 8b 2d b3 0c 21 00 	mov    0x210cb3(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  40995d:	ba 05 00 00 00       	mov    $0x5,%edx
  409962:	31 ff                	xor    %edi,%edi
  409964:	be 28 4c 41 00       	mov    $0x414c28,%esi
  409969:	e8 f2 89 ff ff       	callq  402360 <dcgettext@plt>
  40996e:	48 89 ee             	mov    %rbp,%rsi
  409971:	48 89 c7             	mov    %rax,%rdi
  409974:	e8 a7 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  409979:	48 8b 2d 90 0c 21 00 	mov    0x210c90(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409980:	ba 05 00 00 00       	mov    $0x5,%edx
  409985:	31 ff                	xor    %edi,%edi
  409987:	be a0 4d 41 00       	mov    $0x414da0,%esi
  40998c:	e8 cf 89 ff ff       	callq  402360 <dcgettext@plt>
  409991:	48 89 ee             	mov    %rbp,%rsi
  409994:	48 89 c7             	mov    %rax,%rdi
  409997:	e8 84 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  40999c:	48 8b 2d 6d 0c 21 00 	mov    0x210c6d(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4099a3:	ba 05 00 00 00       	mov    $0x5,%edx
  4099a8:	31 ff                	xor    %edi,%edi
  4099aa:	be 58 4f 41 00       	mov    $0x414f58,%esi
  4099af:	e8 ac 89 ff ff       	callq  402360 <dcgettext@plt>
  4099b4:	48 89 ee             	mov    %rbp,%rsi
  4099b7:	48 89 c7             	mov    %rax,%rdi
  4099ba:	e8 61 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  4099bf:	48 8b 2d 4a 0c 21 00 	mov    0x210c4a(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4099c6:	ba 05 00 00 00       	mov    $0x5,%edx
  4099cb:	31 ff                	xor    %edi,%edi
  4099cd:	be 20 50 41 00       	mov    $0x415020,%esi
  4099d2:	e8 89 89 ff ff       	callq  402360 <dcgettext@plt>
  4099d7:	48 89 ee             	mov    %rbp,%rsi
  4099da:	48 89 c7             	mov    %rax,%rdi
  4099dd:	e8 3e 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  4099e2:	48 8b 2d 27 0c 21 00 	mov    0x210c27(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  4099e9:	ba 05 00 00 00       	mov    $0x5,%edx
  4099ee:	31 ff                	xor    %edi,%edi
  4099f0:	be f0 51 41 00       	mov    $0x4151f0,%esi
  4099f5:	e8 66 89 ff ff       	callq  402360 <dcgettext@plt>
  4099fa:	48 89 ee             	mov    %rbp,%rsi
  4099fd:	48 89 c7             	mov    %rax,%rdi
  409a00:	e8 1b 8b ff ff       	callq  402520 <fputs_unlocked@plt>
  409a05:	48 8b 2d 04 0c 21 00 	mov    0x210c04(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409a0c:	ba 05 00 00 00       	mov    $0x5,%edx
  409a11:	31 ff                	xor    %edi,%edi
  409a13:	be f0 53 41 00       	mov    $0x4153f0,%esi
  409a18:	e8 43 89 ff ff       	callq  402360 <dcgettext@plt>
  409a1d:	48 89 ee             	mov    %rbp,%rsi
  409a20:	48 89 c7             	mov    %rax,%rdi
  409a23:	e8 f8 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409a28:	48 8b 2d e1 0b 21 00 	mov    0x210be1(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409a2f:	ba 05 00 00 00       	mov    $0x5,%edx
  409a34:	31 ff                	xor    %edi,%edi
  409a36:	be 80 54 41 00       	mov    $0x415480,%esi
  409a3b:	e8 20 89 ff ff       	callq  402360 <dcgettext@plt>
  409a40:	48 89 ee             	mov    %rbp,%rsi
  409a43:	48 89 c7             	mov    %rax,%rdi
  409a46:	e8 d5 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409a4b:	48 8b 2d be 0b 21 00 	mov    0x210bbe(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409a52:	ba 05 00 00 00       	mov    $0x5,%edx
  409a57:	31 ff                	xor    %edi,%edi
  409a59:	be e8 55 41 00       	mov    $0x4155e8,%esi
  409a5e:	e8 fd 88 ff ff       	callq  402360 <dcgettext@plt>
  409a63:	48 89 ee             	mov    %rbp,%rsi
  409a66:	48 89 c7             	mov    %rax,%rdi
  409a69:	e8 b2 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409a6e:	48 8b 2d 9b 0b 21 00 	mov    0x210b9b(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409a75:	ba 05 00 00 00       	mov    $0x5,%edx
  409a7a:	31 ff                	xor    %edi,%edi
  409a7c:	be 48 57 41 00       	mov    $0x415748,%esi
  409a81:	e8 da 88 ff ff       	callq  402360 <dcgettext@plt>
  409a86:	48 89 ee             	mov    %rbp,%rsi
  409a89:	48 89 c7             	mov    %rax,%rdi
  409a8c:	e8 8f 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409a91:	48 8b 2d 78 0b 21 00 	mov    0x210b78(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409a98:	ba 05 00 00 00       	mov    $0x5,%edx
  409a9d:	31 ff                	xor    %edi,%edi
  409a9f:	be 78 57 41 00       	mov    $0x415778,%esi
  409aa4:	e8 b7 88 ff ff       	callq  402360 <dcgettext@plt>
  409aa9:	48 89 ee             	mov    %rbp,%rsi
  409aac:	48 89 c7             	mov    %rax,%rdi
  409aaf:	e8 6c 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409ab4:	48 8b 2d 55 0b 21 00 	mov    0x210b55(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409abb:	ba 05 00 00 00       	mov    $0x5,%edx
  409ac0:	31 ff                	xor    %edi,%edi
  409ac2:	be b0 57 41 00       	mov    $0x4157b0,%esi
  409ac7:	e8 94 88 ff ff       	callq  402360 <dcgettext@plt>
  409acc:	48 89 ee             	mov    %rbp,%rsi
  409acf:	48 89 c7             	mov    %rax,%rdi
  409ad2:	e8 49 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409ad7:	48 8b 2d 32 0b 21 00 	mov    0x210b32(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409ade:	ba 05 00 00 00       	mov    $0x5,%edx
  409ae3:	31 ff                	xor    %edi,%edi
  409ae5:	be 50 58 41 00       	mov    $0x415850,%esi
  409aea:	e8 71 88 ff ff       	callq  402360 <dcgettext@plt>
  409aef:	48 89 ee             	mov    %rbp,%rsi
  409af2:	48 89 c7             	mov    %rax,%rdi
  409af5:	e8 26 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409afa:	48 8b 2d 0f 0b 21 00 	mov    0x210b0f(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409b01:	ba 05 00 00 00       	mov    $0x5,%edx
  409b06:	31 ff                	xor    %edi,%edi
  409b08:	be 70 59 41 00       	mov    $0x415970,%esi
  409b0d:	e8 4e 88 ff ff       	callq  402360 <dcgettext@plt>
  409b12:	48 89 ee             	mov    %rbp,%rsi
  409b15:	48 89 c7             	mov    %rax,%rdi
  409b18:	e8 03 8a ff ff       	callq  402520 <fputs_unlocked@plt>
  409b1d:	48 8b 3d dc 16 21 00 	mov    0x2116dc(%rip),%rdi        # 61b200 <stderr@@GLIBC_2.2.5+0xbb0>
  409b24:	e8 67 08 00 00       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  409b29:	31 ff                	xor    %edi,%edi
  409b2b:	48 89 c5             	mov    %rax,%rbp
  409b2e:	ba 05 00 00 00       	mov    $0x5,%edx
  409b33:	be bb 37 41 00       	mov    $0x4137bb,%esi
  409b38:	e8 23 88 ff ff       	callq  402360 <dcgettext@plt>
  409b3d:	b9 d2 37 41 00       	mov    $0x4137d2,%ecx
  409b42:	48 89 ea             	mov    %rbp,%rdx
  409b45:	48 89 c6             	mov    %rax,%rsi
  409b48:	bf 01 00 00 00       	mov    $0x1,%edi
  409b4d:	31 c0                	xor    %eax,%eax
  409b4f:	e8 dc 8b ff ff       	callq  402730 <__printf_chk@plt>
  409b54:	31 ff                	xor    %edi,%edi
  409b56:	ba 05 00 00 00       	mov    $0x5,%edx
  409b5b:	be e8 37 41 00       	mov    $0x4137e8,%esi
  409b60:	e8 fb 87 ff ff       	callq  402360 <dcgettext@plt>
  409b65:	b9 08 5a 41 00       	mov    $0x415a08,%ecx
  409b6a:	48 89 c6             	mov    %rax,%rsi
  409b6d:	ba fc 37 41 00       	mov    $0x4137fc,%edx
  409b72:	bf 01 00 00 00       	mov    $0x1,%edi
  409b77:	31 c0                	xor    %eax,%eax
  409b79:	e8 b2 8b ff ff       	callq  402730 <__printf_chk@plt>
  409b7e:	48 8b 2d 8b 0a 21 00 	mov    0x210a8b(%rip),%rbp        # 61a610 <stdout@@GLIBC_2.2.5>
  409b85:	ba 05 00 00 00       	mov    $0x5,%edx
  409b8a:	31 ff                	xor    %edi,%edi
  409b8c:	be 30 5a 41 00       	mov    $0x415a30,%esi
  409b91:	e8 ca 87 ff ff       	callq  402360 <dcgettext@plt>
  409b96:	48 89 ee             	mov    %rbp,%rsi
  409b99:	48 89 c7             	mov    %rax,%rdi
  409b9c:	e8 7f 89 ff ff       	callq  402520 <fputs_unlocked@plt>
  409ba1:	31 f6                	xor    %esi,%esi
  409ba3:	bf 05 00 00 00       	mov    $0x5,%edi
  409ba8:	e8 63 8b ff ff       	callq  402710 <setlocale@plt>
  409bad:	48 85 c0             	test   %rax,%rax
  409bb0:	74 16                	je     409bc8 <__sprintf_chk@plt+0x7338>
  409bb2:	ba 03 00 00 00       	mov    $0x3,%edx
  409bb7:	be 0a 38 41 00       	mov    $0x41380a,%esi
  409bbc:	48 89 c7             	mov    %rax,%rdi
  409bbf:	e8 7c 86 ff ff       	callq  402240 <strncmp@plt>
  409bc4:	85 c0                	test   %eax,%eax
  409bc6:	75 37                	jne    409bff <__sprintf_chk@plt+0x736f>
  409bc8:	48 8b 3d 31 16 21 00 	mov    0x211631(%rip),%rdi        # 61b200 <stderr@@GLIBC_2.2.5+0xbb0>
  409bcf:	e8 bc 07 00 00       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  409bd4:	31 ff                	xor    %edi,%edi
  409bd6:	48 89 c5             	mov    %rax,%rbp
  409bd9:	ba 05 00 00 00       	mov    $0x5,%edx
  409bde:	be b8 5a 41 00       	mov    $0x415ab8,%esi
  409be3:	e8 78 87 ff ff       	callq  402360 <dcgettext@plt>
  409be8:	48 89 ea             	mov    %rbp,%rdx
  409beb:	48 89 c6             	mov    %rax,%rsi
  409bee:	bf 01 00 00 00       	mov    $0x1,%edi
  409bf3:	31 c0                	xor    %eax,%eax
  409bf5:	e8 36 8b ff ff       	callq  402730 <__printf_chk@plt>
  409bfa:	e9 8e fb ff ff       	jmpq   40978d <__sprintf_chk@plt+0x6efd>
  409bff:	48 8b 3d fa 15 21 00 	mov    0x2115fa(%rip),%rdi        # 61b200 <stderr@@GLIBC_2.2.5+0xbb0>
  409c06:	e8 85 07 00 00       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  409c0b:	31 ff                	xor    %edi,%edi
  409c0d:	48 89 c5             	mov    %rax,%rbp
  409c10:	ba 05 00 00 00       	mov    $0x5,%edx
  409c15:	be 70 5a 41 00       	mov    $0x415a70,%esi
  409c1a:	e8 41 87 ff ff       	callq  402360 <dcgettext@plt>
  409c1f:	48 89 ea             	mov    %rbp,%rdx
  409c22:	48 89 c6             	mov    %rax,%rsi
  409c25:	bf 01 00 00 00       	mov    $0x1,%edi
  409c2a:	31 c0                	xor    %eax,%eax
  409c2c:	e8 ff 8a ff ff       	callq  402730 <__printf_chk@plt>
  409c31:	eb 95                	jmp    409bc8 <__sprintf_chk@plt+0x7338>
  409c33:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  409c3a:	00 00 00 
  409c3d:	0f 1f 00             	nopl   (%rax)
  409c40:	53                   	push   %rbx
  409c41:	31 f6                	xor    %esi,%esi
  409c43:	48 89 fb             	mov    %rdi,%rbx
  409c46:	48 83 ec 10          	sub    $0x10,%rsp
  409c4a:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  409c4f:	e8 7c 8a ff ff       	callq  4026d0 <acl_get_entry@plt>
  409c54:	85 c0                	test   %eax,%eax
  409c56:	7f 31                	jg     409c89 <__sprintf_chk@plt+0x73f9>
  409c58:	eb 47                	jmp    409ca1 <__sprintf_chk@plt+0x7411>
  409c5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409c60:	8b 44 24 04          	mov    0x4(%rsp),%eax
  409c64:	83 f8 01             	cmp    $0x1,%eax
  409c67:	74 0a                	je     409c73 <__sprintf_chk@plt+0x73e3>
  409c69:	83 f8 04             	cmp    $0x4,%eax
  409c6c:	74 05                	je     409c73 <__sprintf_chk@plt+0x73e3>
  409c6e:	83 f8 20             	cmp    $0x20,%eax
  409c71:	75 3d                	jne    409cb0 <__sprintf_chk@plt+0x7420>
  409c73:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  409c78:	be 01 00 00 00       	mov    $0x1,%esi
  409c7d:	48 89 df             	mov    %rbx,%rdi
  409c80:	e8 4b 8a ff ff       	callq  4026d0 <acl_get_entry@plt>
  409c85:	85 c0                	test   %eax,%eax
  409c87:	7e 18                	jle    409ca1 <__sprintf_chk@plt+0x7411>
  409c89:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  409c8e:	48 8d 74 24 04       	lea    0x4(%rsp),%rsi
  409c93:	e8 f8 87 ff ff       	callq  402490 <acl_get_tag_type@plt>
  409c98:	85 c0                	test   %eax,%eax
  409c9a:	79 c4                	jns    409c60 <__sprintf_chk@plt+0x73d0>
  409c9c:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  409ca1:	48 83 c4 10          	add    $0x10,%rsp
  409ca5:	5b                   	pop    %rbx
  409ca6:	c3                   	retq   
  409ca7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  409cae:	00 00 
  409cb0:	48 83 c4 10          	add    $0x10,%rsp
  409cb4:	b8 01 00 00 00       	mov    $0x1,%eax
  409cb9:	5b                   	pop    %rbx
  409cba:	c3                   	retq   
  409cbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409cc0:	8b 46 18             	mov    0x18(%rsi),%eax
  409cc3:	25 00 f0 00 00       	and    $0xf000,%eax
  409cc8:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  409ccd:	74 41                	je     409d10 <__sprintf_chk@plt+0x7480>
  409ccf:	48 83 ec 08          	sub    $0x8,%rsp
  409cd3:	e8 f8 87 ff ff       	callq  4024d0 <acl_extended_file@plt>
  409cd8:	85 c0                	test   %eax,%eax
  409cda:	78 0c                	js     409ce8 <__sprintf_chk@plt+0x7458>
  409cdc:	48 83 c4 08          	add    $0x8,%rsp
  409ce0:	c3                   	retq   
  409ce1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409ce8:	e8 43 85 ff ff       	callq  402230 <__errno_location@plt>
  409ced:	8b 00                	mov    (%rax),%eax
  409cef:	83 f8 5f             	cmp    $0x5f,%eax
  409cf2:	74 24                	je     409d18 <__sprintf_chk@plt+0x7488>
  409cf4:	83 f8 26             	cmp    $0x26,%eax
  409cf7:	74 1f                	je     409d18 <__sprintf_chk@plt+0x7488>
  409cf9:	83 f8 16             	cmp    $0x16,%eax
  409cfc:	74 1a                	je     409d18 <__sprintf_chk@plt+0x7488>
  409cfe:	83 f8 10             	cmp    $0x10,%eax
  409d01:	0f 95 c0             	setne  %al
  409d04:	0f b6 c0             	movzbl %al,%eax
  409d07:	f7 d8                	neg    %eax
  409d09:	eb d1                	jmp    409cdc <__sprintf_chk@plt+0x744c>
  409d0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409d10:	31 c0                	xor    %eax,%eax
  409d12:	c3                   	retq   
  409d13:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409d18:	31 c0                	xor    %eax,%eax
  409d1a:	48 83 c4 08          	add    $0x8,%rsp
  409d1e:	c3                   	retq   
  409d1f:	90                   	nop
  409d20:	41 57                	push   %r15
  409d22:	48 8d 46 01          	lea    0x1(%rsi),%rax
  409d26:	49 bf fe ff ff ff ff 	movabs $0x7ffffffffffffffe,%r15
  409d2d:	ff ff 7f 
  409d30:	41 56                	push   %r14
  409d32:	49 be ff ff ff ff ff 	movabs $0x3fffffffffffffff,%r14
  409d39:	ff ff 3f 
  409d3c:	41 55                	push   %r13
  409d3e:	49 89 fd             	mov    %rdi,%r13
  409d41:	41 54                	push   %r12
  409d43:	55                   	push   %rbp
  409d44:	53                   	push   %rbx
  409d45:	bb 01 04 00 00       	mov    $0x401,%ebx
  409d4a:	48 83 ec 18          	sub    $0x18,%rsp
  409d4e:	48 81 fe 00 04 00 00 	cmp    $0x400,%rsi
  409d55:	48 0f 46 d8          	cmovbe %rax,%rbx
  409d59:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409d60:	48 89 df             	mov    %rbx,%rdi
  409d63:	e8 d8 88 ff ff       	callq  402640 <malloc@plt>
  409d68:	48 85 c0             	test   %rax,%rax
  409d6b:	48 89 c5             	mov    %rax,%rbp
  409d6e:	74 3b                	je     409dab <__sprintf_chk@plt+0x751b>
  409d70:	48 89 da             	mov    %rbx,%rdx
  409d73:	48 89 c6             	mov    %rax,%rsi
  409d76:	4c 89 ef             	mov    %r13,%rdi
  409d79:	e8 62 85 ff ff       	callq  4022e0 <readlink@plt>
  409d7e:	48 85 c0             	test   %rax,%rax
  409d81:	49 89 c4             	mov    %rax,%r12
  409d84:	78 52                	js     409dd8 <__sprintf_chk@plt+0x7548>
  409d86:	4c 39 e3             	cmp    %r12,%rbx
  409d89:	77 7d                	ja     409e08 <__sprintf_chk@plt+0x7578>
  409d8b:	48 89 ef             	mov    %rbp,%rdi
  409d8e:	e8 5d 84 ff ff       	callq  4021f0 <free@plt>
  409d93:	4c 39 f3             	cmp    %r14,%rbx
  409d96:	77 28                	ja     409dc0 <__sprintf_chk@plt+0x7530>
  409d98:	48 01 db             	add    %rbx,%rbx
  409d9b:	48 89 df             	mov    %rbx,%rdi
  409d9e:	e8 9d 88 ff ff       	callq  402640 <malloc@plt>
  409da3:	48 85 c0             	test   %rax,%rax
  409da6:	48 89 c5             	mov    %rax,%rbp
  409da9:	75 c5                	jne    409d70 <__sprintf_chk@plt+0x74e0>
  409dab:	31 c0                	xor    %eax,%eax
  409dad:	48 83 c4 18          	add    $0x18,%rsp
  409db1:	5b                   	pop    %rbx
  409db2:	5d                   	pop    %rbp
  409db3:	41 5c                	pop    %r12
  409db5:	41 5d                	pop    %r13
  409db7:	41 5e                	pop    %r14
  409db9:	41 5f                	pop    %r15
  409dbb:	c3                   	retq   
  409dbc:	0f 1f 40 00          	nopl   0x0(%rax)
  409dc0:	4c 39 fb             	cmp    %r15,%rbx
  409dc3:	77 5b                	ja     409e20 <__sprintf_chk@plt+0x7590>
  409dc5:	48 bb ff ff ff ff ff 	movabs $0x7fffffffffffffff,%rbx
  409dcc:	ff ff 7f 
  409dcf:	eb 8f                	jmp    409d60 <__sprintf_chk@plt+0x74d0>
  409dd1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  409dd8:	e8 53 84 ff ff       	callq  402230 <__errno_location@plt>
  409ddd:	8b 10                	mov    (%rax),%edx
  409ddf:	83 fa 22             	cmp    $0x22,%edx
  409de2:	74 a2                	je     409d86 <__sprintf_chk@plt+0x74f6>
  409de4:	48 89 ef             	mov    %rbp,%rdi
  409de7:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  409deb:	48 89 04 24          	mov    %rax,(%rsp)
  409def:	e8 fc 83 ff ff       	callq  4021f0 <free@plt>
  409df4:	48 8b 04 24          	mov    (%rsp),%rax
  409df8:	8b 54 24 0c          	mov    0xc(%rsp),%edx
  409dfc:	89 10                	mov    %edx,(%rax)
  409dfe:	31 c0                	xor    %eax,%eax
  409e00:	eb ab                	jmp    409dad <__sprintf_chk@plt+0x751d>
  409e02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409e08:	42 c6 44 25 00 00    	movb   $0x0,0x0(%rbp,%r12,1)
  409e0e:	48 83 c4 18          	add    $0x18,%rsp
  409e12:	48 89 e8             	mov    %rbp,%rax
  409e15:	5b                   	pop    %rbx
  409e16:	5d                   	pop    %rbp
  409e17:	41 5c                	pop    %r12
  409e19:	41 5d                	pop    %r13
  409e1b:	41 5e                	pop    %r14
  409e1d:	41 5f                	pop    %r15
  409e1f:	c3                   	retq   
  409e20:	e8 0b 84 ff ff       	callq  402230 <__errno_location@plt>
  409e25:	c7 00 0c 00 00 00    	movl   $0xc,(%rax)
  409e2b:	48 83 c4 18          	add    $0x18,%rsp
  409e2f:	31 c0                	xor    %eax,%eax
  409e31:	5b                   	pop    %rbx
  409e32:	5d                   	pop    %rbp
  409e33:	41 5c                	pop    %r12
  409e35:	41 5d                	pop    %r13
  409e37:	41 5e                	pop    %r14
  409e39:	41 5f                	pop    %r15
  409e3b:	c3                   	retq   
  409e3c:	0f 1f 40 00          	nopl   0x0(%rax)
  409e40:	bf 01 00 00 00       	mov    $0x1,%edi
  409e45:	e9 06 f9 ff ff       	jmpq   409750 <__sprintf_chk@plt+0x6ec0>
  409e4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409e50:	41 57                	push   %r15
  409e52:	49 89 f7             	mov    %rsi,%r15
  409e55:	41 56                	push   %r14
  409e57:	41 55                	push   %r13
  409e59:	41 54                	push   %r12
  409e5b:	49 89 cc             	mov    %rcx,%r12
  409e5e:	55                   	push   %rbp
  409e5f:	48 89 d5             	mov    %rdx,%rbp
  409e62:	53                   	push   %rbx
  409e63:	48 83 ec 28          	sub    $0x28,%rsp
  409e67:	48 89 3c 24          	mov    %rdi,(%rsp)
  409e6b:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  409e70:	e8 0b 85 ff ff       	callq  402380 <strlen@plt>
  409e75:	4d 8b 37             	mov    (%r15),%r14
  409e78:	4d 85 f6             	test   %r14,%r14
  409e7b:	0f 84 f1 00 00 00    	je     409f72 <__sprintf_chk@plt+0x76e2>
  409e81:	49 89 c5             	mov    %rax,%r13
  409e84:	c6 44 24 17 00       	movb   $0x0,0x17(%rsp)
  409e89:	48 c7 44 24 08 ff ff 	movq   $0xffffffffffffffff,0x8(%rsp)
  409e90:	ff ff 
  409e92:	31 db                	xor    %ebx,%ebx
  409e94:	eb 52                	jmp    409ee8 <__sprintf_chk@plt+0x7658>
  409e96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  409e9d:	00 00 00 
  409ea0:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  409ea5:	48 85 c0             	test   %rax,%rax
  409ea8:	0f 84 a2 00 00 00    	je     409f50 <__sprintf_chk@plt+0x76c0>
  409eae:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
  409eb3:	4c 89 e2             	mov    %r12,%rdx
  409eb6:	48 89 ee             	mov    %rbp,%rsi
  409eb9:	49 0f af fc          	imul   %r12,%rdi
  409ebd:	48 01 c7             	add    %rax,%rdi
  409ec0:	e8 3b 86 ff ff       	callq  402500 <memcmp@plt>
  409ec5:	0f b6 4c 24 17       	movzbl 0x17(%rsp),%ecx
  409eca:	85 c0                	test   %eax,%eax
  409ecc:	b8 01 00 00 00       	mov    $0x1,%eax
  409ed1:	0f 45 c8             	cmovne %eax,%ecx
  409ed4:	88 4c 24 17          	mov    %cl,0x17(%rsp)
  409ed8:	48 83 c3 01          	add    $0x1,%rbx
  409edc:	4c 01 e5             	add    %r12,%rbp
  409edf:	4d 8b 34 df          	mov    (%r15,%rbx,8),%r14
  409ee3:	4d 85 f6             	test   %r14,%r14
  409ee6:	74 40                	je     409f28 <__sprintf_chk@plt+0x7698>
  409ee8:	48 8b 34 24          	mov    (%rsp),%rsi
  409eec:	4c 89 ea             	mov    %r13,%rdx
  409eef:	4c 89 f7             	mov    %r14,%rdi
  409ef2:	e8 49 83 ff ff       	callq  402240 <strncmp@plt>
  409ef7:	85 c0                	test   %eax,%eax
  409ef9:	75 dd                	jne    409ed8 <__sprintf_chk@plt+0x7648>
  409efb:	4c 89 f7             	mov    %r14,%rdi
  409efe:	e8 7d 84 ff ff       	callq  402380 <strlen@plt>
  409f03:	4c 39 e8             	cmp    %r13,%rax
  409f06:	74 58                	je     409f60 <__sprintf_chk@plt+0x76d0>
  409f08:	48 83 7c 24 08 ff    	cmpq   $0xffffffffffffffff,0x8(%rsp)
  409f0e:	75 90                	jne    409ea0 <__sprintf_chk@plt+0x7610>
  409f10:	48 89 5c 24 08       	mov    %rbx,0x8(%rsp)
  409f15:	48 83 c3 01          	add    $0x1,%rbx
  409f19:	4c 01 e5             	add    %r12,%rbp
  409f1c:	4d 8b 34 df          	mov    (%r15,%rbx,8),%r14
  409f20:	4d 85 f6             	test   %r14,%r14
  409f23:	75 c3                	jne    409ee8 <__sprintf_chk@plt+0x7658>
  409f25:	0f 1f 00             	nopl   (%rax)
  409f28:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
  409f2d:	48 c7 c0 fe ff ff ff 	mov    $0xfffffffffffffffe,%rax
  409f34:	75 05                	jne    409f3b <__sprintf_chk@plt+0x76ab>
  409f36:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  409f3b:	48 83 c4 28          	add    $0x28,%rsp
  409f3f:	5b                   	pop    %rbx
  409f40:	5d                   	pop    %rbp
  409f41:	41 5c                	pop    %r12
  409f43:	41 5d                	pop    %r13
  409f45:	41 5e                	pop    %r14
  409f47:	41 5f                	pop    %r15
  409f49:	c3                   	retq   
  409f4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  409f50:	c6 44 24 17 01       	movb   $0x1,0x17(%rsp)
  409f55:	eb 81                	jmp    409ed8 <__sprintf_chk@plt+0x7648>
  409f57:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  409f5e:	00 00 
  409f60:	48 83 c4 28          	add    $0x28,%rsp
  409f64:	48 89 d8             	mov    %rbx,%rax
  409f67:	5b                   	pop    %rbx
  409f68:	5d                   	pop    %rbp
  409f69:	41 5c                	pop    %r12
  409f6b:	41 5d                	pop    %r13
  409f6d:	41 5e                	pop    %r14
  409f6f:	41 5f                	pop    %r15
  409f71:	c3                   	retq   
  409f72:	48 c7 44 24 08 ff ff 	movq   $0xffffffffffffffff,0x8(%rsp)
  409f79:	ff ff 
  409f7b:	eb b9                	jmp    409f36 <__sprintf_chk@plt+0x76a6>
  409f7d:	0f 1f 00             	nopl   (%rax)
  409f80:	41 54                	push   %r12
  409f82:	48 83 fa ff          	cmp    $0xffffffffffffffff,%rdx
  409f86:	ba 05 00 00 00       	mov    $0x5,%edx
  409f8b:	55                   	push   %rbp
  409f8c:	48 89 fd             	mov    %rdi,%rbp
  409f8f:	53                   	push   %rbx
  409f90:	48 89 f3             	mov    %rsi,%rbx
  409f93:	74 4b                	je     409fe0 <__sprintf_chk@plt+0x7750>
  409f95:	be 18 5e 41 00       	mov    $0x415e18,%esi
  409f9a:	31 ff                	xor    %edi,%edi
  409f9c:	e8 bf 83 ff ff       	callq  402360 <dcgettext@plt>
  409fa1:	49 89 c4             	mov    %rax,%r12
  409fa4:	48 89 ee             	mov    %rbp,%rsi
  409fa7:	bf 01 00 00 00       	mov    $0x1,%edi
  409fac:	e8 3f 4c 00 00       	callq  40ebf0 <__sprintf_chk@plt+0xc360>
  409fb1:	48 89 da             	mov    %rbx,%rdx
  409fb4:	be 06 00 00 00       	mov    $0x6,%esi
  409fb9:	31 ff                	xor    %edi,%edi
  409fbb:	48 89 c5             	mov    %rax,%rbp
  409fbe:	e8 ad 49 00 00       	callq  40e970 <__sprintf_chk@plt+0xc0e0>
  409fc3:	5b                   	pop    %rbx
  409fc4:	49 89 e8             	mov    %rbp,%r8
  409fc7:	4c 89 e2             	mov    %r12,%rdx
  409fca:	48 89 c1             	mov    %rax,%rcx
  409fcd:	5d                   	pop    %rbp
  409fce:	41 5c                	pop    %r12
  409fd0:	31 f6                	xor    %esi,%esi
  409fd2:	31 ff                	xor    %edi,%edi
  409fd4:	31 c0                	xor    %eax,%eax
  409fd6:	e9 95 87 ff ff       	jmpq   402770 <error@plt>
  409fdb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  409fe0:	be fd 5d 41 00       	mov    $0x415dfd,%esi
  409fe5:	31 ff                	xor    %edi,%edi
  409fe7:	e8 74 83 ff ff       	callq  402360 <dcgettext@plt>
  409fec:	49 89 c4             	mov    %rax,%r12
  409fef:	eb b3                	jmp    409fa4 <__sprintf_chk@plt+0x7714>
  409ff1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  409ff8:	0f 1f 84 00 00 00 00 
  409fff:	00 
  40a000:	41 57                	push   %r15
  40a002:	49 89 ff             	mov    %rdi,%r15
  40a005:	31 ff                	xor    %edi,%edi
  40a007:	41 56                	push   %r14
  40a009:	45 31 f6             	xor    %r14d,%r14d
  40a00c:	41 55                	push   %r13
  40a00e:	49 89 d5             	mov    %rdx,%r13
  40a011:	ba 05 00 00 00       	mov    $0x5,%edx
  40a016:	41 54                	push   %r12
  40a018:	55                   	push   %rbp
  40a019:	48 89 f5             	mov    %rsi,%rbp
  40a01c:	be 35 5e 41 00       	mov    $0x415e35,%esi
  40a021:	53                   	push   %rbx
  40a022:	48 83 ec 08          	sub    $0x8,%rsp
  40a026:	48 8b 1d 23 06 21 00 	mov    0x210623(%rip),%rbx        # 61a650 <stderr@@GLIBC_2.2.5>
  40a02d:	e8 2e 83 ff ff       	callq  402360 <dcgettext@plt>
  40a032:	48 89 c7             	mov    %rax,%rdi
  40a035:	48 89 de             	mov    %rbx,%rsi
  40a038:	31 db                	xor    %ebx,%ebx
  40a03a:	e8 e1 84 ff ff       	callq  402520 <fputs_unlocked@plt>
  40a03f:	4d 8b 27             	mov    (%r15),%r12
  40a042:	4d 85 e4             	test   %r12,%r12
  40a045:	75 3f                	jne    40a086 <__sprintf_chk@plt+0x77f6>
  40a047:	e9 84 00 00 00       	jmpq   40a0d0 <__sprintf_chk@plt+0x7840>
  40a04c:	0f 1f 40 00          	nopl   0x0(%rax)
  40a050:	4c 89 e7             	mov    %r12,%rdi
  40a053:	48 83 c3 01          	add    $0x1,%rbx
  40a057:	49 89 ee             	mov    %rbp,%r14
  40a05a:	e8 b1 4b 00 00       	callq  40ec10 <__sprintf_chk@plt+0xc380>
  40a05f:	48 8b 3d ea 05 21 00 	mov    0x2105ea(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  40a066:	48 89 c1             	mov    %rax,%rcx
  40a069:	ba 4a 5e 41 00       	mov    $0x415e4a,%edx
  40a06e:	31 c0                	xor    %eax,%eax
  40a070:	be 01 00 00 00       	mov    $0x1,%esi
  40a075:	4c 01 ed             	add    %r13,%rbp
  40a078:	e8 93 87 ff ff       	callq  402810 <__fprintf_chk@plt>
  40a07d:	4d 8b 24 df          	mov    (%r15,%rbx,8),%r12
  40a081:	4d 85 e4             	test   %r12,%r12
  40a084:	74 4a                	je     40a0d0 <__sprintf_chk@plt+0x7840>
  40a086:	48 85 db             	test   %rbx,%rbx
  40a089:	74 c5                	je     40a050 <__sprintf_chk@plt+0x77c0>
  40a08b:	4c 89 ea             	mov    %r13,%rdx
  40a08e:	48 89 ee             	mov    %rbp,%rsi
  40a091:	4c 89 f7             	mov    %r14,%rdi
  40a094:	e8 67 84 ff ff       	callq  402500 <memcmp@plt>
  40a099:	85 c0                	test   %eax,%eax
  40a09b:	75 b3                	jne    40a050 <__sprintf_chk@plt+0x77c0>
  40a09d:	4c 89 e7             	mov    %r12,%rdi
  40a0a0:	48 83 c3 01          	add    $0x1,%rbx
  40a0a4:	4c 01 ed             	add    %r13,%rbp
  40a0a7:	e8 64 4b 00 00       	callq  40ec10 <__sprintf_chk@plt+0xc380>
  40a0ac:	48 8b 3d 9d 05 21 00 	mov    0x21059d(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  40a0b3:	48 89 c1             	mov    %rax,%rcx
  40a0b6:	ba 52 5e 41 00       	mov    $0x415e52,%edx
  40a0bb:	31 c0                	xor    %eax,%eax
  40a0bd:	be 01 00 00 00       	mov    $0x1,%esi
  40a0c2:	e8 49 87 ff ff       	callq  402810 <__fprintf_chk@plt>
  40a0c7:	4d 8b 24 df          	mov    (%r15,%rbx,8),%r12
  40a0cb:	4d 85 e4             	test   %r12,%r12
  40a0ce:	75 b6                	jne    40a086 <__sprintf_chk@plt+0x77f6>
  40a0d0:	48 8b 3d 79 05 21 00 	mov    0x210579(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  40a0d7:	48 8b 47 28          	mov    0x28(%rdi),%rax
  40a0db:	48 3b 47 30          	cmp    0x30(%rdi),%rax
  40a0df:	73 1a                	jae    40a0fb <__sprintf_chk@plt+0x786b>
  40a0e1:	48 8d 50 01          	lea    0x1(%rax),%rdx
  40a0e5:	48 89 57 28          	mov    %rdx,0x28(%rdi)
  40a0e9:	c6 00 0a             	movb   $0xa,(%rax)
  40a0ec:	48 83 c4 08          	add    $0x8,%rsp
  40a0f0:	5b                   	pop    %rbx
  40a0f1:	5d                   	pop    %rbp
  40a0f2:	41 5c                	pop    %r12
  40a0f4:	41 5d                	pop    %r13
  40a0f6:	41 5e                	pop    %r14
  40a0f8:	41 5f                	pop    %r15
  40a0fa:	c3                   	retq   
  40a0fb:	48 83 c4 08          	add    $0x8,%rsp
  40a0ff:	be 0a 00 00 00       	mov    $0xa,%esi
  40a104:	5b                   	pop    %rbx
  40a105:	5d                   	pop    %rbp
  40a106:	41 5c                	pop    %r12
  40a108:	41 5d                	pop    %r13
  40a10a:	41 5e                	pop    %r14
  40a10c:	41 5f                	pop    %r15
  40a10e:	e9 ed 82 ff ff       	jmpq   402400 <__overflow@plt>
  40a113:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40a11a:	84 00 00 00 00 00 
  40a120:	41 57                	push   %r15
  40a122:	4d 89 c7             	mov    %r8,%r15
  40a125:	41 56                	push   %r14
  40a127:	49 89 fe             	mov    %rdi,%r14
  40a12a:	41 55                	push   %r13
  40a12c:	4d 89 cd             	mov    %r9,%r13
  40a12f:	41 54                	push   %r12
  40a131:	49 89 f4             	mov    %rsi,%r12
  40a134:	4c 89 e7             	mov    %r12,%rdi
  40a137:	55                   	push   %rbp
  40a138:	48 89 cd             	mov    %rcx,%rbp
  40a13b:	4c 89 c1             	mov    %r8,%rcx
  40a13e:	53                   	push   %rbx
  40a13f:	48 89 d3             	mov    %rdx,%rbx
  40a142:	48 89 ea             	mov    %rbp,%rdx
  40a145:	48 89 de             	mov    %rbx,%rsi
  40a148:	48 83 ec 08          	sub    $0x8,%rsp
  40a14c:	e8 ff fc ff ff       	callq  409e50 <__sprintf_chk@plt+0x75c0>
  40a151:	48 85 c0             	test   %rax,%rax
  40a154:	78 0f                	js     40a165 <__sprintf_chk@plt+0x78d5>
  40a156:	48 83 c4 08          	add    $0x8,%rsp
  40a15a:	5b                   	pop    %rbx
  40a15b:	5d                   	pop    %rbp
  40a15c:	41 5c                	pop    %r12
  40a15e:	41 5d                	pop    %r13
  40a160:	41 5e                	pop    %r14
  40a162:	41 5f                	pop    %r15
  40a164:	c3                   	retq   
  40a165:	48 89 c2             	mov    %rax,%rdx
  40a168:	4c 89 e6             	mov    %r12,%rsi
  40a16b:	4c 89 f7             	mov    %r14,%rdi
  40a16e:	e8 0d fe ff ff       	callq  409f80 <__sprintf_chk@plt+0x76f0>
  40a173:	4c 89 fa             	mov    %r15,%rdx
  40a176:	48 89 ee             	mov    %rbp,%rsi
  40a179:	48 89 df             	mov    %rbx,%rdi
  40a17c:	e8 7f fe ff ff       	callq  40a000 <__sprintf_chk@plt+0x7770>
  40a181:	41 ff d5             	callq  *%r13
  40a184:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  40a18b:	eb c9                	jmp    40a156 <__sprintf_chk@plt+0x78c6>
  40a18d:	0f 1f 00             	nopl   (%rax)
  40a190:	41 56                	push   %r14
  40a192:	41 55                	push   %r13
  40a194:	41 54                	push   %r12
  40a196:	55                   	push   %rbp
  40a197:	53                   	push   %rbx
  40a198:	4c 8b 26             	mov    (%rsi),%r12
  40a19b:	4d 85 e4             	test   %r12,%r12
  40a19e:	74 32                	je     40a1d2 <__sprintf_chk@plt+0x7942>
  40a1a0:	49 89 fe             	mov    %rdi,%r14
  40a1a3:	49 89 cd             	mov    %rcx,%r13
  40a1a6:	48 89 d3             	mov    %rdx,%rbx
  40a1a9:	48 8d 6e 08          	lea    0x8(%rsi),%rbp
  40a1ad:	eb 11                	jmp    40a1c0 <__sprintf_chk@plt+0x7930>
  40a1af:	90                   	nop
  40a1b0:	4c 8b 65 00          	mov    0x0(%rbp),%r12
  40a1b4:	4c 01 eb             	add    %r13,%rbx
  40a1b7:	48 83 c5 08          	add    $0x8,%rbp
  40a1bb:	4d 85 e4             	test   %r12,%r12
  40a1be:	74 12                	je     40a1d2 <__sprintf_chk@plt+0x7942>
  40a1c0:	4c 89 ea             	mov    %r13,%rdx
  40a1c3:	48 89 de             	mov    %rbx,%rsi
  40a1c6:	4c 89 f7             	mov    %r14,%rdi
  40a1c9:	e8 32 83 ff ff       	callq  402500 <memcmp@plt>
  40a1ce:	85 c0                	test   %eax,%eax
  40a1d0:	75 de                	jne    40a1b0 <__sprintf_chk@plt+0x7920>
  40a1d2:	5b                   	pop    %rbx
  40a1d3:	5d                   	pop    %rbp
  40a1d4:	4c 89 e0             	mov    %r12,%rax
  40a1d7:	41 5c                	pop    %r12
  40a1d9:	41 5d                	pop    %r13
  40a1db:	41 5e                	pop    %r14
  40a1dd:	c3                   	retq   
  40a1de:	66 90                	xchg   %ax,%ax
  40a1e0:	48 89 3d f1 0f 21 00 	mov    %rdi,0x210ff1(%rip)        # 61b1d8 <stderr@@GLIBC_2.2.5+0xb88>
  40a1e7:	c3                   	retq   
  40a1e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a1ef:	00 
  40a1f0:	40 88 3d d9 0f 21 00 	mov    %dil,0x210fd9(%rip)        # 61b1d0 <stderr@@GLIBC_2.2.5+0xb80>
  40a1f7:	c3                   	retq   
  40a1f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a1ff:	00 
  40a200:	55                   	push   %rbp
  40a201:	53                   	push   %rbx
  40a202:	48 83 ec 08          	sub    $0x8,%rsp
  40a206:	48 8b 3d 03 04 21 00 	mov    0x210403(%rip),%rdi        # 61a610 <stdout@@GLIBC_2.2.5>
  40a20d:	e8 6e 76 00 00       	callq  411880 <__sprintf_chk@plt+0xeff0>
  40a212:	85 c0                	test   %eax,%eax
  40a214:	74 13                	je     40a229 <__sprintf_chk@plt+0x7999>
  40a216:	80 3d b3 0f 21 00 00 	cmpb   $0x0,0x210fb3(%rip)        # 61b1d0 <stderr@@GLIBC_2.2.5+0xb80>
  40a21d:	74 21                	je     40a240 <__sprintf_chk@plt+0x79b0>
  40a21f:	e8 0c 80 ff ff       	callq  402230 <__errno_location@plt>
  40a224:	83 38 20             	cmpl   $0x20,(%rax)
  40a227:	75 17                	jne    40a240 <__sprintf_chk@plt+0x79b0>
  40a229:	48 8b 3d 20 04 21 00 	mov    0x210420(%rip),%rdi        # 61a650 <stderr@@GLIBC_2.2.5>
  40a230:	e8 4b 76 00 00       	callq  411880 <__sprintf_chk@plt+0xeff0>
  40a235:	85 c0                	test   %eax,%eax
  40a237:	75 4a                	jne    40a283 <__sprintf_chk@plt+0x79f3>
  40a239:	48 83 c4 08          	add    $0x8,%rsp
  40a23d:	5b                   	pop    %rbx
  40a23e:	5d                   	pop    %rbp
  40a23f:	c3                   	retq   
  40a240:	31 ff                	xor    %edi,%edi
  40a242:	ba 05 00 00 00       	mov    $0x5,%edx
  40a247:	be 57 5e 41 00       	mov    $0x415e57,%esi
  40a24c:	e8 0f 81 ff ff       	callq  402360 <dcgettext@plt>
  40a251:	48 8b 3d 80 0f 21 00 	mov    0x210f80(%rip),%rdi        # 61b1d8 <stderr@@GLIBC_2.2.5+0xb88>
  40a258:	48 89 c3             	mov    %rax,%rbx
  40a25b:	48 85 ff             	test   %rdi,%rdi
  40a25e:	74 2e                	je     40a28e <__sprintf_chk@plt+0x79fe>
  40a260:	e8 4b 48 00 00       	callq  40eab0 <__sprintf_chk@plt+0xc220>
  40a265:	48 89 c5             	mov    %rax,%rbp
  40a268:	e8 c3 7f ff ff       	callq  402230 <__errno_location@plt>
  40a26d:	8b 30                	mov    (%rax),%esi
  40a26f:	49 89 d8             	mov    %rbx,%r8
  40a272:	48 89 e9             	mov    %rbp,%rcx
  40a275:	ba 63 5e 41 00       	mov    $0x415e63,%edx
  40a27a:	31 ff                	xor    %edi,%edi
  40a27c:	31 c0                	xor    %eax,%eax
  40a27e:	e8 ed 84 ff ff       	callq  402770 <error@plt>
  40a283:	8b 3d f7 02 21 00    	mov    0x2102f7(%rip),%edi        # 61a580 <_fini@@Base+0x208684>
  40a289:	e8 c2 7f ff ff       	callq  402250 <_exit@plt>
  40a28e:	e8 9d 7f ff ff       	callq  402230 <__errno_location@plt>
  40a293:	8b 30                	mov    (%rax),%esi
  40a295:	48 89 d9             	mov    %rbx,%rcx
  40a298:	ba 54 5e 41 00       	mov    $0x415e54,%edx
  40a29d:	31 ff                	xor    %edi,%edi
  40a29f:	31 c0                	xor    %eax,%eax
  40a2a1:	e8 ca 84 ff ff       	callq  402770 <error@plt>
  40a2a6:	eb db                	jmp    40a283 <__sprintf_chk@plt+0x79f3>
  40a2a8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a2af:	00 
  40a2b0:	55                   	push   %rbp
  40a2b1:	31 ed                	xor    %ebp,%ebp
  40a2b3:	53                   	push   %rbx
  40a2b4:	48 89 fb             	mov    %rdi,%rbx
  40a2b7:	48 83 ec 08          	sub    $0x8,%rsp
  40a2bb:	80 3f 2f             	cmpb   $0x2f,(%rdi)
  40a2be:	40 0f 94 c5          	sete   %bpl
  40a2c2:	e8 c9 00 00 00       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  40a2c7:	48 89 c1             	mov    %rax,%rcx
  40a2ca:	48 29 d9             	sub    %rbx,%rcx
  40a2cd:	48 39 e9             	cmp    %rbp,%rcx
  40a2d0:	76 2b                	jbe    40a2fd <__sprintf_chk@plt+0x7a6d>
  40a2d2:	80 78 ff 2f          	cmpb   $0x2f,-0x1(%rax)
  40a2d6:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
  40a2da:	74 12                	je     40a2ee <__sprintf_chk@plt+0x7a5e>
  40a2dc:	eb 1f                	jmp    40a2fd <__sprintf_chk@plt+0x7a6d>
  40a2de:	66 90                	xchg   %ax,%ax
  40a2e0:	80 7c 13 ff 2f       	cmpb   $0x2f,-0x1(%rbx,%rdx,1)
  40a2e5:	48 8d 42 ff          	lea    -0x1(%rdx),%rax
  40a2e9:	75 08                	jne    40a2f3 <__sprintf_chk@plt+0x7a63>
  40a2eb:	48 89 c2             	mov    %rax,%rdx
  40a2ee:	48 39 d5             	cmp    %rdx,%rbp
  40a2f1:	72 ed                	jb     40a2e0 <__sprintf_chk@plt+0x7a50>
  40a2f3:	48 83 c4 08          	add    $0x8,%rsp
  40a2f7:	48 89 d0             	mov    %rdx,%rax
  40a2fa:	5b                   	pop    %rbx
  40a2fb:	5d                   	pop    %rbp
  40a2fc:	c3                   	retq   
  40a2fd:	48 89 ca             	mov    %rcx,%rdx
  40a300:	eb f1                	jmp    40a2f3 <__sprintf_chk@plt+0x7a63>
  40a302:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40a309:	1f 84 00 00 00 00 00 
  40a310:	41 55                	push   %r13
  40a312:	41 54                	push   %r12
  40a314:	49 89 fc             	mov    %rdi,%r12
  40a317:	55                   	push   %rbp
  40a318:	53                   	push   %rbx
  40a319:	48 83 ec 08          	sub    $0x8,%rsp
  40a31d:	e8 8e ff ff ff       	callq  40a2b0 <__sprintf_chk@plt+0x7a20>
  40a322:	48 85 c0             	test   %rax,%rax
  40a325:	4c 8d 68 01          	lea    0x1(%rax),%r13
  40a329:	48 89 c3             	mov    %rax,%rbx
  40a32c:	40 0f 94 c5          	sete   %bpl
  40a330:	40 0f b6 fd          	movzbl %bpl,%edi
  40a334:	4c 01 ef             	add    %r13,%rdi
  40a337:	e8 04 83 ff ff       	callq  402640 <malloc@plt>
  40a33c:	48 85 c0             	test   %rax,%rax
  40a33f:	74 3f                	je     40a380 <__sprintf_chk@plt+0x7af0>
  40a341:	48 89 da             	mov    %rbx,%rdx
  40a344:	4c 89 e6             	mov    %r12,%rsi
  40a347:	48 89 c7             	mov    %rax,%rdi
  40a34a:	e8 71 82 ff ff       	callq  4025c0 <memcpy@plt>
  40a34f:	40 84 ed             	test   %bpl,%bpl
  40a352:	48 89 c1             	mov    %rax,%rcx
  40a355:	75 19                	jne    40a370 <__sprintf_chk@plt+0x7ae0>
  40a357:	c6 04 19 00          	movb   $0x0,(%rcx,%rbx,1)
  40a35b:	48 89 c8             	mov    %rcx,%rax
  40a35e:	48 83 c4 08          	add    $0x8,%rsp
  40a362:	5b                   	pop    %rbx
  40a363:	5d                   	pop    %rbp
  40a364:	41 5c                	pop    %r12
  40a366:	41 5d                	pop    %r13
  40a368:	c3                   	retq   
  40a369:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40a370:	c6 04 18 2e          	movb   $0x2e,(%rax,%rbx,1)
  40a374:	4c 89 eb             	mov    %r13,%rbx
  40a377:	eb de                	jmp    40a357 <__sprintf_chk@plt+0x7ac7>
  40a379:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40a380:	31 c0                	xor    %eax,%eax
  40a382:	eb da                	jmp    40a35e <__sprintf_chk@plt+0x7ace>
  40a384:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40a38b:	00 00 00 
  40a38e:	66 90                	xchg   %ax,%ax
  40a390:	0f b6 17             	movzbl (%rdi),%edx
  40a393:	48 89 f8             	mov    %rdi,%rax
  40a396:	80 fa 2f             	cmp    $0x2f,%dl
  40a399:	75 11                	jne    40a3ac <__sprintf_chk@plt+0x7b1c>
  40a39b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40a3a0:	48 83 c0 01          	add    $0x1,%rax
  40a3a4:	0f b6 10             	movzbl (%rax),%edx
  40a3a7:	80 fa 2f             	cmp    $0x2f,%dl
  40a3aa:	74 f4                	je     40a3a0 <__sprintf_chk@plt+0x7b10>
  40a3ac:	89 d1                	mov    %edx,%ecx
  40a3ae:	31 f6                	xor    %esi,%esi
  40a3b0:	48 89 c2             	mov    %rax,%rdx
  40a3b3:	84 c9                	test   %cl,%cl
  40a3b5:	74 40                	je     40a3f7 <__sprintf_chk@plt+0x7b67>
  40a3b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40a3be:	00 00 
  40a3c0:	48 83 c2 01          	add    $0x1,%rdx
  40a3c4:	0f b6 0a             	movzbl (%rdx),%ecx
  40a3c7:	84 c9                	test   %cl,%cl
  40a3c9:	74 1a                	je     40a3e5 <__sprintf_chk@plt+0x7b55>
  40a3cb:	80 f9 2f             	cmp    $0x2f,%cl
  40a3ce:	74 20                	je     40a3f0 <__sprintf_chk@plt+0x7b60>
  40a3d0:	40 84 f6             	test   %sil,%sil
  40a3d3:	74 eb                	je     40a3c0 <__sprintf_chk@plt+0x7b30>
  40a3d5:	48 89 d0             	mov    %rdx,%rax
  40a3d8:	48 83 c2 01          	add    $0x1,%rdx
  40a3dc:	0f b6 0a             	movzbl (%rdx),%ecx
  40a3df:	31 f6                	xor    %esi,%esi
  40a3e1:	84 c9                	test   %cl,%cl
  40a3e3:	75 e6                	jne    40a3cb <__sprintf_chk@plt+0x7b3b>
  40a3e5:	f3 c3                	repz retq 
  40a3e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40a3ee:	00 00 
  40a3f0:	be 01 00 00 00       	mov    $0x1,%esi
  40a3f5:	eb c9                	jmp    40a3c0 <__sprintf_chk@plt+0x7b30>
  40a3f7:	f3 c3                	repz retq 
  40a3f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40a400:	53                   	push   %rbx
  40a401:	48 89 fb             	mov    %rdi,%rbx
  40a404:	e8 77 7f ff ff       	callq  402380 <strlen@plt>
  40a409:	48 83 f8 01          	cmp    $0x1,%rax
  40a40d:	76 0b                	jbe    40a41a <__sprintf_chk@plt+0x7b8a>
  40a40f:	80 7c 03 ff 2f       	cmpb   $0x2f,-0x1(%rbx,%rax,1)
  40a414:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
  40a418:	74 06                	je     40a420 <__sprintf_chk@plt+0x7b90>
  40a41a:	5b                   	pop    %rbx
  40a41b:	c3                   	retq   
  40a41c:	0f 1f 40 00          	nopl   0x0(%rax)
  40a420:	48 83 fa 01          	cmp    $0x1,%rdx
  40a424:	48 89 d0             	mov    %rdx,%rax
  40a427:	75 e6                	jne    40a40f <__sprintf_chk@plt+0x7b7f>
  40a429:	5b                   	pop    %rbx
  40a42a:	c3                   	retq   
  40a42b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40a430:	89 f8                	mov    %edi,%eax
  40a432:	25 00 f0 00 00       	and    $0xf000,%eax
  40a437:	3d 00 80 00 00       	cmp    $0x8000,%eax
  40a43c:	0f 84 6e 01 00 00    	je     40a5b0 <__sprintf_chk@plt+0x7d20>
  40a442:	3d 00 40 00 00       	cmp    $0x4000,%eax
  40a447:	0f 84 73 01 00 00    	je     40a5c0 <__sprintf_chk@plt+0x7d30>
  40a44d:	3d 00 60 00 00       	cmp    $0x6000,%eax
  40a452:	0f 84 78 01 00 00    	je     40a5d0 <__sprintf_chk@plt+0x7d40>
  40a458:	3d 00 20 00 00       	cmp    $0x2000,%eax
  40a45d:	0f 84 3d 01 00 00    	je     40a5a0 <__sprintf_chk@plt+0x7d10>
  40a463:	3d 00 a0 00 00       	cmp    $0xa000,%eax
  40a468:	0f 84 72 01 00 00    	je     40a5e0 <__sprintf_chk@plt+0x7d50>
  40a46e:	3d 00 10 00 00       	cmp    $0x1000,%eax
  40a473:	0f 84 77 01 00 00    	je     40a5f0 <__sprintf_chk@plt+0x7d60>
  40a479:	3d 00 c0 00 00       	cmp    $0xc000,%eax
  40a47e:	ba 73 00 00 00       	mov    $0x73,%edx
  40a483:	b8 3f 00 00 00       	mov    $0x3f,%eax
  40a488:	0f 45 d0             	cmovne %eax,%edx
  40a48b:	89 f8                	mov    %edi,%eax
  40a48d:	88 16                	mov    %dl,(%rsi)
  40a48f:	25 00 01 00 00       	and    $0x100,%eax
  40a494:	83 f8 01             	cmp    $0x1,%eax
  40a497:	19 c0                	sbb    %eax,%eax
  40a499:	83 e0 bb             	and    $0xffffffbb,%eax
  40a49c:	83 c0 72             	add    $0x72,%eax
  40a49f:	88 46 01             	mov    %al,0x1(%rsi)
  40a4a2:	89 f8                	mov    %edi,%eax
  40a4a4:	25 80 00 00 00       	and    $0x80,%eax
  40a4a9:	83 f8 01             	cmp    $0x1,%eax
  40a4ac:	19 c0                	sbb    %eax,%eax
  40a4ae:	83 e0 b6             	and    $0xffffffb6,%eax
  40a4b1:	83 c0 77             	add    $0x77,%eax
  40a4b4:	88 46 02             	mov    %al,0x2(%rsi)
  40a4b7:	89 f8                	mov    %edi,%eax
  40a4b9:	83 e0 40             	and    $0x40,%eax
  40a4bc:	83 f8 01             	cmp    $0x1,%eax
  40a4bf:	19 c0                	sbb    %eax,%eax
  40a4c1:	f7 c7 00 08 00 00    	test   $0x800,%edi
  40a4c7:	0f 84 c3 00 00 00    	je     40a590 <__sprintf_chk@plt+0x7d00>
  40a4cd:	83 e0 e0             	and    $0xffffffe0,%eax
  40a4d0:	83 c0 73             	add    $0x73,%eax
  40a4d3:	88 46 03             	mov    %al,0x3(%rsi)
  40a4d6:	89 f8                	mov    %edi,%eax
  40a4d8:	83 e0 20             	and    $0x20,%eax
  40a4db:	83 f8 01             	cmp    $0x1,%eax
  40a4de:	19 c0                	sbb    %eax,%eax
  40a4e0:	83 e0 bb             	and    $0xffffffbb,%eax
  40a4e3:	83 c0 72             	add    $0x72,%eax
  40a4e6:	88 46 04             	mov    %al,0x4(%rsi)
  40a4e9:	89 f8                	mov    %edi,%eax
  40a4eb:	83 e0 10             	and    $0x10,%eax
  40a4ee:	83 f8 01             	cmp    $0x1,%eax
  40a4f1:	19 c0                	sbb    %eax,%eax
  40a4f3:	83 e0 b6             	and    $0xffffffb6,%eax
  40a4f6:	83 c0 77             	add    $0x77,%eax
  40a4f9:	88 46 05             	mov    %al,0x5(%rsi)
  40a4fc:	89 f8                	mov    %edi,%eax
  40a4fe:	83 e0 08             	and    $0x8,%eax
  40a501:	83 f8 01             	cmp    $0x1,%eax
  40a504:	19 c0                	sbb    %eax,%eax
  40a506:	f7 c7 00 04 00 00    	test   $0x400,%edi
  40a50c:	74 72                	je     40a580 <__sprintf_chk@plt+0x7cf0>
  40a50e:	83 e0 e0             	and    $0xffffffe0,%eax
  40a511:	83 c0 73             	add    $0x73,%eax
  40a514:	88 46 06             	mov    %al,0x6(%rsi)
  40a517:	89 f8                	mov    %edi,%eax
  40a519:	83 e0 04             	and    $0x4,%eax
  40a51c:	83 f8 01             	cmp    $0x1,%eax
  40a51f:	19 c0                	sbb    %eax,%eax
  40a521:	83 e0 bb             	and    $0xffffffbb,%eax
  40a524:	83 c0 72             	add    $0x72,%eax
  40a527:	88 46 07             	mov    %al,0x7(%rsi)
  40a52a:	89 f8                	mov    %edi,%eax
  40a52c:	83 e0 02             	and    $0x2,%eax
  40a52f:	83 f8 01             	cmp    $0x1,%eax
  40a532:	19 c0                	sbb    %eax,%eax
  40a534:	83 e0 b6             	and    $0xffffffb6,%eax
  40a537:	83 c0 77             	add    $0x77,%eax
  40a53a:	f7 c7 00 02 00 00    	test   $0x200,%edi
  40a540:	88 46 08             	mov    %al,0x8(%rsi)
  40a543:	74 1b                	je     40a560 <__sprintf_chk@plt+0x7cd0>
  40a545:	83 e7 01             	and    $0x1,%edi
  40a548:	c6 46 0a 20          	movb   $0x20,0xa(%rsi)
  40a54c:	c6 46 0b 00          	movb   $0x0,0xb(%rsi)
  40a550:	83 ff 01             	cmp    $0x1,%edi
  40a553:	19 c0                	sbb    %eax,%eax
  40a555:	83 e0 e0             	and    $0xffffffe0,%eax
  40a558:	83 c0 74             	add    $0x74,%eax
  40a55b:	88 46 09             	mov    %al,0x9(%rsi)
  40a55e:	c3                   	retq   
  40a55f:	90                   	nop
  40a560:	83 e7 01             	and    $0x1,%edi
  40a563:	c6 46 0a 20          	movb   $0x20,0xa(%rsi)
  40a567:	c6 46 0b 00          	movb   $0x0,0xb(%rsi)
  40a56b:	83 ff 01             	cmp    $0x1,%edi
  40a56e:	19 c0                	sbb    %eax,%eax
  40a570:	83 e0 b5             	and    $0xffffffb5,%eax
  40a573:	83 c0 78             	add    $0x78,%eax
  40a576:	88 46 09             	mov    %al,0x9(%rsi)
  40a579:	c3                   	retq   
  40a57a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a580:	83 e0 b5             	and    $0xffffffb5,%eax
  40a583:	83 c0 78             	add    $0x78,%eax
  40a586:	eb 8c                	jmp    40a514 <__sprintf_chk@plt+0x7c84>
  40a588:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a58f:	00 
  40a590:	83 e0 b5             	and    $0xffffffb5,%eax
  40a593:	83 c0 78             	add    $0x78,%eax
  40a596:	e9 38 ff ff ff       	jmpq   40a4d3 <__sprintf_chk@plt+0x7c43>
  40a59b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40a5a0:	ba 63 00 00 00       	mov    $0x63,%edx
  40a5a5:	e9 e1 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a5b0:	ba 2d 00 00 00       	mov    $0x2d,%edx
  40a5b5:	e9 d1 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a5c0:	ba 64 00 00 00       	mov    $0x64,%edx
  40a5c5:	e9 c1 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a5d0:	ba 62 00 00 00       	mov    $0x62,%edx
  40a5d5:	e9 b1 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a5e0:	ba 6c 00 00 00       	mov    $0x6c,%edx
  40a5e5:	e9 a1 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a5f0:	ba 70 00 00 00       	mov    $0x70,%edx
  40a5f5:	e9 91 fe ff ff       	jmpq   40a48b <__sprintf_chk@plt+0x7bfb>
  40a5fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a600:	8b 7f 18             	mov    0x18(%rdi),%edi
  40a603:	e9 28 fe ff ff       	jmpq   40a430 <__sprintf_chk@plt+0x7ba0>
  40a608:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a60f:	00 
  40a610:	48 83 ec 08          	sub    $0x8,%rsp
  40a614:	e8 17 00 00 00       	callq  40a630 <__sprintf_chk@plt+0x7da0>
  40a619:	48 85 c0             	test   %rax,%rax
  40a61c:	74 05                	je     40a623 <__sprintf_chk@plt+0x7d93>
  40a61e:	48 83 c4 08          	add    $0x8,%rsp
  40a622:	c3                   	retq   
  40a623:	e8 28 68 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  40a628:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a62f:	00 
  40a630:	41 57                	push   %r15
  40a632:	49 89 d7             	mov    %rdx,%r15
  40a635:	41 56                	push   %r14
  40a637:	45 31 f6             	xor    %r14d,%r14d
  40a63a:	41 55                	push   %r13
  40a63c:	41 54                	push   %r12
  40a63e:	49 89 fc             	mov    %rdi,%r12
  40a641:	55                   	push   %rbp
  40a642:	48 89 f5             	mov    %rsi,%rbp
  40a645:	53                   	push   %rbx
  40a646:	48 83 ec 18          	sub    $0x18,%rsp
  40a64a:	e8 41 fd ff ff       	callq  40a390 <__sprintf_chk@plt+0x7b00>
  40a64f:	48 89 c3             	mov    %rax,%rbx
  40a652:	48 89 c7             	mov    %rax,%rdi
  40a655:	e8 a6 fd ff ff       	callq  40a400 <__sprintf_chk@plt+0x7b70>
  40a65a:	48 89 da             	mov    %rbx,%rdx
  40a65d:	4c 29 e2             	sub    %r12,%rdx
  40a660:	48 85 c0             	test   %rax,%rax
  40a663:	48 8d 34 02          	lea    (%rdx,%rax,1),%rsi
  40a667:	48 89 34 24          	mov    %rsi,(%rsp)
  40a66b:	74 0c                	je     40a679 <__sprintf_chk@plt+0x7de9>
  40a66d:	45 31 f6             	xor    %r14d,%r14d
  40a670:	80 7c 03 ff 2f       	cmpb   $0x2f,-0x1(%rbx,%rax,1)
  40a675:	41 0f 95 c6          	setne  %r14b
  40a679:	80 7d 00 2f          	cmpb   $0x2f,0x0(%rbp)
  40a67d:	48 89 eb             	mov    %rbp,%rbx
  40a680:	74 7e                	je     40a700 <__sprintf_chk@plt+0x7e70>
  40a682:	48 89 df             	mov    %rbx,%rdi
  40a685:	e8 f6 7c ff ff       	callq  402380 <strlen@plt>
  40a68a:	48 8b 0c 24          	mov    (%rsp),%rcx
  40a68e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40a693:	4a 8d 7c 31 01       	lea    0x1(%rcx,%r14,1),%rdi
  40a698:	48 01 c7             	add    %rax,%rdi
  40a69b:	e8 a0 7f ff ff       	callq  402640 <malloc@plt>
  40a6a0:	48 85 c0             	test   %rax,%rax
  40a6a3:	49 89 c5             	mov    %rax,%r13
  40a6a6:	74 78                	je     40a720 <__sprintf_chk@plt+0x7e90>
  40a6a8:	48 8b 14 24          	mov    (%rsp),%rdx
  40a6ac:	48 89 c7             	mov    %rax,%rdi
  40a6af:	4c 89 e6             	mov    %r12,%rsi
  40a6b2:	e8 99 80 ff ff       	callq  402750 <mempcpy@plt>
  40a6b7:	4d 85 ff             	test   %r15,%r15
  40a6ba:	c6 00 2f             	movb   $0x2f,(%rax)
  40a6bd:	4a 8d 3c 30          	lea    (%rax,%r14,1),%rdi
  40a6c1:	74 12                	je     40a6d5 <__sprintf_chk@plt+0x7e45>
  40a6c3:	31 c0                	xor    %eax,%eax
  40a6c5:	80 7d 00 2f          	cmpb   $0x2f,0x0(%rbp)
  40a6c9:	48 89 f9             	mov    %rdi,%rcx
  40a6cc:	0f 94 c0             	sete   %al
  40a6cf:	48 29 c1             	sub    %rax,%rcx
  40a6d2:	49 89 0f             	mov    %rcx,(%r15)
  40a6d5:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  40a6da:	48 89 de             	mov    %rbx,%rsi
  40a6dd:	e8 6e 80 ff ff       	callq  402750 <mempcpy@plt>
  40a6e2:	c6 00 00             	movb   $0x0,(%rax)
  40a6e5:	4c 89 e8             	mov    %r13,%rax
  40a6e8:	48 83 c4 18          	add    $0x18,%rsp
  40a6ec:	5b                   	pop    %rbx
  40a6ed:	5d                   	pop    %rbp
  40a6ee:	41 5c                	pop    %r12
  40a6f0:	41 5d                	pop    %r13
  40a6f2:	41 5e                	pop    %r14
  40a6f4:	41 5f                	pop    %r15
  40a6f6:	c3                   	retq   
  40a6f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40a6fe:	00 00 
  40a700:	48 83 c3 01          	add    $0x1,%rbx
  40a704:	80 3b 2f             	cmpb   $0x2f,(%rbx)
  40a707:	0f 85 75 ff ff ff    	jne    40a682 <__sprintf_chk@plt+0x7df2>
  40a70d:	48 83 c3 01          	add    $0x1,%rbx
  40a711:	80 3b 2f             	cmpb   $0x2f,(%rbx)
  40a714:	0f 85 68 ff ff ff    	jne    40a682 <__sprintf_chk@plt+0x7df2>
  40a71a:	eb e4                	jmp    40a700 <__sprintf_chk@plt+0x7e70>
  40a71c:	0f 1f 40 00          	nopl   0x0(%rax)
  40a720:	31 c0                	xor    %eax,%eax
  40a722:	eb c4                	jmp    40a6e8 <__sprintf_chk@plt+0x7e58>
  40a724:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40a72b:	00 00 00 
  40a72e:	66 90                	xchg   %ax,%ax
  40a730:	48 8b 17             	mov    (%rdi),%rdx
  40a733:	0f b6 0a             	movzbl (%rdx),%ecx
  40a736:	84 c9                	test   %cl,%cl
  40a738:	0f 84 8f 00 00 00    	je     40a7cd <__sprintf_chk@plt+0x7f3d>
  40a73e:	45 31 c0             	xor    %r8d,%r8d
  40a741:	31 c0                	xor    %eax,%eax
  40a743:	45 31 d2             	xor    %r10d,%r10d
  40a746:	eb 2e                	jmp    40a776 <__sprintf_chk@plt+0x7ee6>
  40a748:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40a74f:	00 
  40a750:	0f be f1             	movsbl %cl,%esi
  40a753:	45 31 c0             	xor    %r8d,%r8d
  40a756:	83 e6 df             	and    $0xffffffdf,%esi
  40a759:	83 ee 41             	sub    $0x41,%esi
  40a75c:	83 fe 19             	cmp    $0x19,%esi
  40a75f:	76 07                	jbe    40a768 <__sprintf_chk@plt+0x7ed8>
  40a761:	80 f9 7e             	cmp    $0x7e,%cl
  40a764:	49 0f 45 c2          	cmovne %r10,%rax
  40a768:	48 83 c2 01          	add    $0x1,%rdx
  40a76c:	48 89 17             	mov    %rdx,(%rdi)
  40a76f:	0f b6 0a             	movzbl (%rdx),%ecx
  40a772:	84 c9                	test   %cl,%cl
  40a774:	74 3a                	je     40a7b0 <__sprintf_chk@plt+0x7f20>
  40a776:	45 84 c0             	test   %r8b,%r8b
  40a779:	75 d5                	jne    40a750 <__sprintf_chk@plt+0x7ec0>
  40a77b:	80 f9 2e             	cmp    $0x2e,%cl
  40a77e:	74 38                	je     40a7b8 <__sprintf_chk@plt+0x7f28>
  40a780:	0f be f1             	movsbl %cl,%esi
  40a783:	44 8d 4e d0          	lea    -0x30(%rsi),%r9d
  40a787:	41 83 f9 09          	cmp    $0x9,%r9d
  40a78b:	76 db                	jbe    40a768 <__sprintf_chk@plt+0x7ed8>
  40a78d:	83 e6 df             	and    $0xffffffdf,%esi
  40a790:	83 ee 41             	sub    $0x41,%esi
  40a793:	83 fe 19             	cmp    $0x19,%esi
  40a796:	77 c9                	ja     40a761 <__sprintf_chk@plt+0x7ed1>
  40a798:	48 83 c2 01          	add    $0x1,%rdx
  40a79c:	48 89 17             	mov    %rdx,(%rdi)
  40a79f:	0f b6 0a             	movzbl (%rdx),%ecx
  40a7a2:	84 c9                	test   %cl,%cl
  40a7a4:	75 d0                	jne    40a776 <__sprintf_chk@plt+0x7ee6>
  40a7a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40a7ad:	00 00 00 
  40a7b0:	f3 c3                	repz retq 
  40a7b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40a7b8:	48 85 c0             	test   %rax,%rax
  40a7bb:	74 0b                	je     40a7c8 <__sprintf_chk@plt+0x7f38>
  40a7bd:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  40a7c3:	eb a3                	jmp    40a768 <__sprintf_chk@plt+0x7ed8>
  40a7c5:	0f 1f 00             	nopl   (%rax)
  40a7c8:	48 89 d0             	mov    %rdx,%rax
  40a7cb:	eb f0                	jmp    40a7bd <__sprintf_chk@plt+0x7f2d>
  40a7cd:	31 c0                	xor    %eax,%eax
  40a7cf:	c3                   	retq   
  40a7d0:	41 56                	push   %r14
  40a7d2:	41 55                	push   %r13
  40a7d4:	41 54                	push   %r12
  40a7d6:	55                   	push   %rbp
  40a7d7:	48 89 f5             	mov    %rsi,%rbp
  40a7da:	53                   	push   %rbx
  40a7db:	48 89 fb             	mov    %rdi,%rbx
  40a7de:	48 83 ec 10          	sub    $0x10,%rsp
  40a7e2:	e8 69 7d ff ff       	callq  402550 <strcmp@plt>
  40a7e7:	41 89 c5             	mov    %eax,%r13d
  40a7ea:	31 c0                	xor    %eax,%eax
  40a7ec:	45 85 ed             	test   %r13d,%r13d
  40a7ef:	74 47                	je     40a838 <__sprintf_chk@plt+0x7fa8>
  40a7f1:	0f b6 13             	movzbl (%rbx),%edx
  40a7f4:	84 d2                	test   %dl,%dl
  40a7f6:	0f 84 eb 02 00 00    	je     40aae7 <__sprintf_chk@plt+0x8257>
  40a7fc:	0f b6 4d 00          	movzbl 0x0(%rbp),%ecx
  40a800:	b0 01                	mov    $0x1,%al
  40a802:	84 c9                	test   %cl,%cl
  40a804:	74 32                	je     40a838 <__sprintf_chk@plt+0x7fa8>
  40a806:	0f b6 c2             	movzbl %dl,%eax
  40a809:	be 2e 00 00 00       	mov    $0x2e,%esi
  40a80e:	29 c6                	sub    %eax,%esi
  40a810:	75 0b                	jne    40a81d <__sprintf_chk@plt+0x7f8d>
  40a812:	80 7b 01 00          	cmpb   $0x0,0x1(%rbx)
  40a816:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40a81b:	74 1b                	je     40a838 <__sprintf_chk@plt+0x7fa8>
  40a81d:	0f b6 c1             	movzbl %cl,%eax
  40a820:	bf 2e 00 00 00       	mov    $0x2e,%edi
  40a825:	29 c7                	sub    %eax,%edi
  40a827:	75 1f                	jne    40a848 <__sprintf_chk@plt+0x7fb8>
  40a829:	80 7d 01 00          	cmpb   $0x0,0x1(%rbp)
  40a82d:	b8 01 00 00 00       	mov    $0x1,%eax
  40a832:	75 14                	jne    40a848 <__sprintf_chk@plt+0x7fb8>
  40a834:	0f 1f 40 00          	nopl   0x0(%rax)
  40a838:	48 83 c4 10          	add    $0x10,%rsp
  40a83c:	5b                   	pop    %rbx
  40a83d:	5d                   	pop    %rbp
  40a83e:	41 5c                	pop    %r12
  40a840:	41 5d                	pop    %r13
  40a842:	41 5e                	pop    %r14
  40a844:	c3                   	retq   
  40a845:	0f 1f 00             	nopl   (%rax)
  40a848:	85 f6                	test   %esi,%esi
  40a84a:	75 11                	jne    40a85d <__sprintf_chk@plt+0x7fcd>
  40a84c:	80 7b 01 2e          	cmpb   $0x2e,0x1(%rbx)
  40a850:	75 0b                	jne    40a85d <__sprintf_chk@plt+0x7fcd>
  40a852:	80 7b 02 00          	cmpb   $0x0,0x2(%rbx)
  40a856:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40a85b:	74 db                	je     40a838 <__sprintf_chk@plt+0x7fa8>
  40a85d:	85 ff                	test   %edi,%edi
  40a85f:	0f 84 3b 01 00 00    	je     40a9a0 <__sprintf_chk@plt+0x8110>
  40a865:	80 fa 2e             	cmp    $0x2e,%dl
  40a868:	0f 84 52 02 00 00    	je     40aac0 <__sprintf_chk@plt+0x8230>
  40a86e:	80 f9 2e             	cmp    $0x2e,%cl
  40a871:	0f 84 3d 01 00 00    	je     40a9b4 <__sprintf_chk@plt+0x8124>
  40a877:	48 89 e7             	mov    %rsp,%rdi
  40a87a:	48 89 1c 24          	mov    %rbx,(%rsp)
  40a87e:	48 89 6c 24 08       	mov    %rbp,0x8(%rsp)
  40a883:	e8 a8 fe ff ff       	callq  40a730 <__sprintf_chk@plt+0x7ea0>
  40a888:	48 8d 7c 24 08       	lea    0x8(%rsp),%rdi
  40a88d:	49 89 c6             	mov    %rax,%r14
  40a890:	4d 89 f4             	mov    %r14,%r12
  40a893:	e8 98 fe ff ff       	callq  40a730 <__sprintf_chk@plt+0x7ea0>
  40a898:	4d 85 f6             	test   %r14,%r14
  40a89b:	4c 0f 44 24 24       	cmove  (%rsp),%r12
  40a8a0:	49 89 c2             	mov    %rax,%r10
  40a8a3:	49 29 ea             	sub    %rbp,%r10
  40a8a6:	49 29 dc             	sub    %rbx,%r12
  40a8a9:	48 85 c0             	test   %rax,%rax
  40a8ac:	0f 84 3f 02 00 00    	je     40aaf1 <__sprintf_chk@plt+0x8261>
  40a8b2:	4d 39 d4             	cmp    %r10,%r12
  40a8b5:	0f 84 d0 01 00 00    	je     40aa8b <__sprintf_chk@plt+0x81fb>
  40a8bb:	31 c9                	xor    %ecx,%ecx
  40a8bd:	45 31 c0             	xor    %r8d,%r8d
  40a8c0:	41 bb ff ff ff ff    	mov    $0xffffffff,%r11d
  40a8c6:	49 39 ca             	cmp    %rcx,%r10
  40a8c9:	0f 87 a2 00 00 00    	ja     40a971 <__sprintf_chk@plt+0x80e1>
  40a8cf:	e9 33 02 00 00       	jmpq   40ab07 <__sprintf_chk@plt+0x8277>
  40a8d4:	0f 1f 40 00          	nopl   0x0(%rax)
  40a8d8:	49 39 ca             	cmp    %rcx,%r10
  40a8db:	44 0f be 4c 0d 00    	movsbl 0x0(%rbp,%rcx,1),%r9d
  40a8e1:	0f 86 ed 00 00 00    	jbe    40a9d4 <__sprintf_chk@plt+0x8144>
  40a8e7:	41 0f be c1          	movsbl %r9b,%eax
  40a8eb:	83 e8 30             	sub    $0x30,%eax
  40a8ee:	83 f8 09             	cmp    $0x9,%eax
  40a8f1:	0f 86 dd 00 00 00    	jbe    40a9d4 <__sprintf_chk@plt+0x8144>
  40a8f7:	4d 39 c4             	cmp    %r8,%r12
  40a8fa:	0f 84 18 02 00 00    	je     40ab18 <__sprintf_chk@plt+0x8288>
  40a900:	42 0f b6 14 03       	movzbl (%rbx,%r8,1),%edx
  40a905:	0f b6 f2             	movzbl %dl,%esi
  40a908:	31 c0                	xor    %eax,%eax
  40a90a:	8d 7e d0             	lea    -0x30(%rsi),%edi
  40a90d:	83 ff 09             	cmp    $0x9,%edi
  40a910:	76 13                	jbe    40a925 <__sprintf_chk@plt+0x8095>
  40a912:	89 f0                	mov    %esi,%eax
  40a914:	83 e0 df             	and    $0xffffffdf,%eax
  40a917:	83 e8 41             	sub    $0x41,%eax
  40a91a:	83 f8 19             	cmp    $0x19,%eax
  40a91d:	0f 87 45 01 00 00    	ja     40aa68 <__sprintf_chk@plt+0x81d8>
  40a923:	89 f0                	mov    %esi,%eax
  40a925:	49 39 ca             	cmp    %rcx,%r10
  40a928:	0f 84 f1 01 00 00    	je     40ab1f <__sprintf_chk@plt+0x828f>
  40a92e:	44 0f b6 4c 0d 00    	movzbl 0x0(%rbp,%rcx,1),%r9d
  40a934:	41 0f b6 f1          	movzbl %r9b,%esi
  40a938:	31 ff                	xor    %edi,%edi
  40a93a:	8d 56 d0             	lea    -0x30(%rsi),%edx
  40a93d:	83 fa 09             	cmp    $0x9,%edx
  40a940:	76 1f                	jbe    40a961 <__sprintf_chk@plt+0x80d1>
  40a942:	89 f2                	mov    %esi,%edx
  40a944:	89 f7                	mov    %esi,%edi
  40a946:	83 e2 df             	and    $0xffffffdf,%edx
  40a949:	83 ea 41             	sub    $0x41,%edx
  40a94c:	83 fa 19             	cmp    $0x19,%edx
  40a94f:	76 10                	jbe    40a961 <__sprintf_chk@plt+0x80d1>
  40a951:	81 c6 00 01 00 00    	add    $0x100,%esi
  40a957:	41 80 f9 7e          	cmp    $0x7e,%r9b
  40a95b:	89 f7                	mov    %esi,%edi
  40a95d:	41 0f 44 fb          	cmove  %r11d,%edi
  40a961:	39 f8                	cmp    %edi,%eax
  40a963:	0f 85 17 01 00 00    	jne    40aa80 <__sprintf_chk@plt+0x81f0>
  40a969:	49 83 c0 01          	add    $0x1,%r8
  40a96d:	48 83 c1 01          	add    $0x1,%rcx
  40a971:	4d 39 c4             	cmp    %r8,%r12
  40a974:	0f 86 5e ff ff ff    	jbe    40a8d8 <__sprintf_chk@plt+0x8048>
  40a97a:	42 0f b6 34 03       	movzbl (%rbx,%r8,1),%esi
  40a97f:	40 0f be c6          	movsbl %sil,%eax
  40a983:	83 e8 30             	sub    $0x30,%eax
  40a986:	83 f8 09             	cmp    $0x9,%eax
  40a989:	0f 86 49 ff ff ff    	jbe    40a8d8 <__sprintf_chk@plt+0x8048>
  40a98f:	89 f2                	mov    %esi,%edx
  40a991:	e9 6f ff ff ff       	jmpq   40a905 <__sprintf_chk@plt+0x8075>
  40a996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40a99d:	00 00 00 
  40a9a0:	80 7d 01 2e          	cmpb   $0x2e,0x1(%rbp)
  40a9a4:	0f 85 bb fe ff ff    	jne    40a865 <__sprintf_chk@plt+0x7fd5>
  40a9aa:	80 7d 02 00          	cmpb   $0x0,0x2(%rbp)
  40a9ae:	0f 85 b1 fe ff ff    	jne    40a865 <__sprintf_chk@plt+0x7fd5>
  40a9b4:	48 83 c4 10          	add    $0x10,%rsp
  40a9b8:	b8 01 00 00 00       	mov    $0x1,%eax
  40a9bd:	5b                   	pop    %rbx
  40a9be:	5d                   	pop    %rbp
  40a9bf:	41 5c                	pop    %r12
  40a9c1:	41 5d                	pop    %r13
  40a9c3:	41 5e                	pop    %r14
  40a9c5:	c3                   	retq   
  40a9c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40a9cd:	00 00 00 
  40a9d0:	49 83 c0 01          	add    $0x1,%r8
  40a9d4:	42 0f b6 14 03       	movzbl (%rbx,%r8,1),%edx
  40a9d9:	80 fa 30             	cmp    $0x30,%dl
  40a9dc:	74 f2                	je     40a9d0 <__sprintf_chk@plt+0x8140>
  40a9de:	eb 0a                	jmp    40a9ea <__sprintf_chk@plt+0x815a>
  40a9e0:	48 83 c1 01          	add    $0x1,%rcx
  40a9e4:	44 0f be 4c 0d 00    	movsbl 0x0(%rbp,%rcx,1),%r9d
  40a9ea:	41 80 f9 30          	cmp    $0x30,%r9b
  40a9ee:	74 f0                	je     40a9e0 <__sprintf_chk@plt+0x8150>
  40a9f0:	0f be c2             	movsbl %dl,%eax
  40a9f3:	83 e8 30             	sub    $0x30,%eax
  40a9f6:	83 f8 09             	cmp    $0x9,%eax
  40a9f9:	41 0f be c1          	movsbl %r9b,%eax
  40a9fd:	0f 87 d8 00 00 00    	ja     40aadb <__sprintf_chk@plt+0x824b>
  40aa03:	83 e8 30             	sub    $0x30,%eax
  40aa06:	83 f8 09             	cmp    $0x9,%eax
  40aa09:	77 a9                	ja     40a9b4 <__sprintf_chk@plt+0x8124>
  40aa0b:	31 c0                	xor    %eax,%eax
  40aa0d:	eb 13                	jmp    40aa22 <__sprintf_chk@plt+0x8192>
  40aa0f:	90                   	nop
  40aa10:	44 0f be 4c 0d 00    	movsbl 0x0(%rbp,%rcx,1),%r9d
  40aa16:	41 0f be f1          	movsbl %r9b,%esi
  40aa1a:	83 ee 30             	sub    $0x30,%esi
  40aa1d:	83 fe 09             	cmp    $0x9,%esi
  40aa20:	77 92                	ja     40a9b4 <__sprintf_chk@plt+0x8124>
  40aa22:	85 c0                	test   %eax,%eax
  40aa24:	75 06                	jne    40aa2c <__sprintf_chk@plt+0x819c>
  40aa26:	0f be c2             	movsbl %dl,%eax
  40aa29:	44 29 c8             	sub    %r9d,%eax
  40aa2c:	49 83 c0 01          	add    $0x1,%r8
  40aa30:	48 83 c1 01          	add    $0x1,%rcx
  40aa34:	42 0f b6 14 03       	movzbl (%rbx,%r8,1),%edx
  40aa39:	0f be f2             	movsbl %dl,%esi
  40aa3c:	83 ee 30             	sub    $0x30,%esi
  40aa3f:	83 fe 09             	cmp    $0x9,%esi
  40aa42:	76 cc                	jbe    40aa10 <__sprintf_chk@plt+0x8180>
  40aa44:	0f be 54 0d 00       	movsbl 0x0(%rbp,%rcx,1),%edx
  40aa49:	83 ea 30             	sub    $0x30,%edx
  40aa4c:	83 fa 09             	cmp    $0x9,%edx
  40aa4f:	0f 86 92 00 00 00    	jbe    40aae7 <__sprintf_chk@plt+0x8257>
  40aa55:	85 c0                	test   %eax,%eax
  40aa57:	0f 84 69 fe ff ff    	je     40a8c6 <__sprintf_chk@plt+0x8036>
  40aa5d:	e9 d6 fd ff ff       	jmpq   40a838 <__sprintf_chk@plt+0x7fa8>
  40aa62:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40aa68:	81 c6 00 01 00 00    	add    $0x100,%esi
  40aa6e:	80 fa 7e             	cmp    $0x7e,%dl
  40aa71:	89 f0                	mov    %esi,%eax
  40aa73:	41 0f 44 c3          	cmove  %r11d,%eax
  40aa77:	e9 a9 fe ff ff       	jmpq   40a925 <__sprintf_chk@plt+0x8095>
  40aa7c:	0f 1f 40 00          	nopl   0x0(%rax)
  40aa80:	29 f8                	sub    %edi,%eax
  40aa82:	41 0f 44 c5          	cmove  %r13d,%eax
  40aa86:	e9 ad fd ff ff       	jmpq   40a838 <__sprintf_chk@plt+0x7fa8>
  40aa8b:	4c 89 e2             	mov    %r12,%rdx
  40aa8e:	48 89 ee             	mov    %rbp,%rsi
  40aa91:	48 89 df             	mov    %rbx,%rdi
  40aa94:	e8 a7 77 ff ff       	callq  402240 <strncmp@plt>
  40aa99:	85 c0                	test   %eax,%eax
  40aa9b:	4d 89 e2             	mov    %r12,%r10
  40aa9e:	0f 85 17 fe ff ff    	jne    40a8bb <__sprintf_chk@plt+0x802b>
  40aaa4:	4c 8b 24 24          	mov    (%rsp),%r12
  40aaa8:	4c 8b 54 24 08       	mov    0x8(%rsp),%r10
  40aaad:	49 29 dc             	sub    %rbx,%r12
  40aab0:	49 29 ea             	sub    %rbp,%r10
  40aab3:	e9 03 fe ff ff       	jmpq   40a8bb <__sprintf_chk@plt+0x802b>
  40aab8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40aabf:	00 
  40aac0:	48 83 c3 01          	add    $0x1,%rbx
  40aac4:	48 83 c5 01          	add    $0x1,%rbp
  40aac8:	80 f9 2e             	cmp    $0x2e,%cl
  40aacb:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40aad0:	0f 84 a1 fd ff ff    	je     40a877 <__sprintf_chk@plt+0x7fe7>
  40aad6:	e9 5d fd ff ff       	jmpq   40a838 <__sprintf_chk@plt+0x7fa8>
  40aadb:	83 e8 30             	sub    $0x30,%eax
  40aade:	83 f8 09             	cmp    $0x9,%eax
  40aae1:	0f 87 df fd ff ff    	ja     40a8c6 <__sprintf_chk@plt+0x8036>
  40aae7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40aaec:	e9 47 fd ff ff       	jmpq   40a838 <__sprintf_chk@plt+0x7fa8>
  40aaf1:	4c 8b 54 24 08       	mov    0x8(%rsp),%r10
  40aaf6:	49 29 ea             	sub    %rbp,%r10
  40aaf9:	4d 85 f6             	test   %r14,%r14
  40aafc:	0f 84 b9 fd ff ff    	je     40a8bb <__sprintf_chk@plt+0x802b>
  40ab02:	e9 ab fd ff ff       	jmpq   40a8b2 <__sprintf_chk@plt+0x8022>
  40ab07:	4d 39 c4             	cmp    %r8,%r12
  40ab0a:	0f 87 61 fe ff ff    	ja     40a971 <__sprintf_chk@plt+0x80e1>
  40ab10:	44 89 e8             	mov    %r13d,%eax
  40ab13:	e9 20 fd ff ff       	jmpq   40a838 <__sprintf_chk@plt+0x7fa8>
  40ab18:	31 c0                	xor    %eax,%eax
  40ab1a:	e9 15 fe ff ff       	jmpq   40a934 <__sprintf_chk@plt+0x80a4>
  40ab1f:	31 ff                	xor    %edi,%edi
  40ab21:	e9 3b fe ff ff       	jmpq   40a961 <__sprintf_chk@plt+0x80d1>
  40ab26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40ab2d:	00 00 00 
  40ab30:	53                   	push   %rbx
  40ab31:	48 89 fe             	mov    %rdi,%rsi
  40ab34:	48 89 fb             	mov    %rdi,%rbx
  40ab37:	31 ff                	xor    %edi,%edi
  40ab39:	48 83 ec 10          	sub    $0x10,%rsp
  40ab3d:	e8 ae 77 ff ff       	callq  4022f0 <clock_gettime@plt>
  40ab42:	85 c0                	test   %eax,%eax
  40ab44:	74 21                	je     40ab67 <__sprintf_chk@plt+0x82d7>
  40ab46:	31 f6                	xor    %esi,%esi
  40ab48:	48 89 e7             	mov    %rsp,%rdi
  40ab4b:	e8 f0 78 ff ff       	callq  402440 <gettimeofday@plt>
  40ab50:	48 8b 04 24          	mov    (%rsp),%rax
  40ab54:	48 89 03             	mov    %rax,(%rbx)
  40ab57:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40ab5c:	48 69 c0 e8 03 00 00 	imul   $0x3e8,%rax,%rax
  40ab63:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40ab67:	48 83 c4 10          	add    $0x10,%rsp
  40ab6b:	5b                   	pop    %rbx
  40ab6c:	c3                   	retq   
  40ab6d:	0f 1f 00             	nopl   (%rax)
  40ab70:	48 83 ec 08          	sub    $0x8,%rsp
  40ab74:	31 f6                	xor    %esi,%esi
  40ab76:	e8 95 7b ff ff       	callq  402710 <setlocale@plt>
  40ab7b:	48 85 c0             	test   %rax,%rax
  40ab7e:	74 30                	je     40abb0 <__sprintf_chk@plt+0x8320>
  40ab80:	80 38 43             	cmpb   $0x43,(%rax)
  40ab83:	75 13                	jne    40ab98 <__sprintf_chk@plt+0x8308>
  40ab85:	80 78 01 00          	cmpb   $0x0,0x1(%rax)
  40ab89:	75 0d                	jne    40ab98 <__sprintf_chk@plt+0x8308>
  40ab8b:	31 c0                	xor    %eax,%eax
  40ab8d:	48 83 c4 08          	add    $0x8,%rsp
  40ab91:	c3                   	retq   
  40ab92:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ab98:	48 89 c6             	mov    %rax,%rsi
  40ab9b:	bf 6a 5e 41 00       	mov    $0x415e6a,%edi
  40aba0:	b9 06 00 00 00       	mov    $0x6,%ecx
  40aba5:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  40aba7:	0f 95 c0             	setne  %al
  40abaa:	48 83 c4 08          	add    $0x8,%rsp
  40abae:	c3                   	retq   
  40abaf:	90                   	nop
  40abb0:	b8 01 00 00 00       	mov    $0x1,%eax
  40abb5:	48 83 c4 08          	add    $0x8,%rsp
  40abb9:	c3                   	retq   
  40abba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40abc0:	48 83 ff 09          	cmp    $0x9,%rdi
  40abc4:	0f 87 85 00 00 00    	ja     40ac4f <__sprintf_chk@plt+0x83bf>
  40abca:	bf 0b 00 00 00       	mov    $0xb,%edi
  40abcf:	49 b9 ab aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaab,%r9
  40abd6:	aa aa aa 
  40abd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40abe0:	48 83 ff 09          	cmp    $0x9,%rdi
  40abe4:	76 62                	jbe    40ac48 <__sprintf_chk@plt+0x83b8>
  40abe6:	48 89 f8             	mov    %rdi,%rax
  40abe9:	49 f7 e1             	mul    %r9
  40abec:	48 d1 ea             	shr    %rdx
  40abef:	48 8d 04 52          	lea    (%rdx,%rdx,2),%rax
  40abf3:	48 39 c7             	cmp    %rax,%rdi
  40abf6:	74 42                	je     40ac3a <__sprintf_chk@plt+0x83aa>
  40abf8:	41 b8 10 00 00 00    	mov    $0x10,%r8d
  40abfe:	be 09 00 00 00       	mov    $0x9,%esi
  40ac03:	b9 03 00 00 00       	mov    $0x3,%ecx
  40ac08:	eb 17                	jmp    40ac21 <__sprintf_chk@plt+0x8391>
  40ac0a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ac10:	31 d2                	xor    %edx,%edx
  40ac12:	48 89 f8             	mov    %rdi,%rax
  40ac15:	49 83 c0 08          	add    $0x8,%r8
  40ac19:	48 f7 f1             	div    %rcx
  40ac1c:	48 85 d2             	test   %rdx,%rdx
  40ac1f:	74 19                	je     40ac3a <__sprintf_chk@plt+0x83aa>
  40ac21:	4c 01 c6             	add    %r8,%rsi
  40ac24:	48 83 c1 02          	add    $0x2,%rcx
  40ac28:	48 39 fe             	cmp    %rdi,%rsi
  40ac2b:	72 e3                	jb     40ac10 <__sprintf_chk@plt+0x8380>
  40ac2d:	31 d2                	xor    %edx,%edx
  40ac2f:	48 89 f8             	mov    %rdi,%rax
  40ac32:	48 f7 f1             	div    %rcx
  40ac35:	48 85 d2             	test   %rdx,%rdx
  40ac38:	75 0a                	jne    40ac44 <__sprintf_chk@plt+0x83b4>
  40ac3a:	48 83 c7 02          	add    $0x2,%rdi
  40ac3e:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  40ac42:	75 9c                	jne    40abe0 <__sprintf_chk@plt+0x8350>
  40ac44:	48 89 f8             	mov    %rdi,%rax
  40ac47:	c3                   	retq   
  40ac48:	b9 03 00 00 00       	mov    $0x3,%ecx
  40ac4d:	eb de                	jmp    40ac2d <__sprintf_chk@plt+0x839d>
  40ac4f:	48 83 cf 01          	or     $0x1,%rdi
  40ac53:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
  40ac57:	0f 85 72 ff ff ff    	jne    40abcf <__sprintf_chk@plt+0x833f>
  40ac5d:	eb e5                	jmp    40ac44 <__sprintf_chk@plt+0x83b4>
  40ac5f:	90                   	nop
  40ac60:	48 c1 cf 03          	ror    $0x3,%rdi
  40ac64:	31 d2                	xor    %edx,%edx
  40ac66:	48 89 f8             	mov    %rdi,%rax
  40ac69:	48 f7 f6             	div    %rsi
  40ac6c:	48 89 d0             	mov    %rdx,%rax
  40ac6f:	c3                   	retq   
  40ac70:	48 39 f7             	cmp    %rsi,%rdi
  40ac73:	0f 94 c0             	sete   %al
  40ac76:	c3                   	retq   
  40ac77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40ac7e:	00 00 
  40ac80:	53                   	push   %rbx
  40ac81:	48 89 fb             	mov    %rdi,%rbx
  40ac84:	48 89 f7             	mov    %rsi,%rdi
  40ac87:	48 8b 73 10          	mov    0x10(%rbx),%rsi
  40ac8b:	ff 53 30             	callq  *0x30(%rbx)
  40ac8e:	48 3b 43 10          	cmp    0x10(%rbx),%rax
  40ac92:	73 09                	jae    40ac9d <__sprintf_chk@plt+0x840d>
  40ac94:	48 c1 e0 04          	shl    $0x4,%rax
  40ac98:	48 03 03             	add    (%rbx),%rax
  40ac9b:	5b                   	pop    %rbx
  40ac9c:	c3                   	retq   
  40ac9d:	e8 7e 75 ff ff       	callq  402220 <abort@plt>
  40aca2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40aca9:	1f 84 00 00 00 00 00 
  40acb0:	41 56                	push   %r14
  40acb2:	49 89 d6             	mov    %rdx,%r14
  40acb5:	41 55                	push   %r13
  40acb7:	41 89 cd             	mov    %ecx,%r13d
  40acba:	41 54                	push   %r12
  40acbc:	49 89 f4             	mov    %rsi,%r12
  40acbf:	55                   	push   %rbp
  40acc0:	48 89 fd             	mov    %rdi,%rbp
  40acc3:	53                   	push   %rbx
  40acc4:	e8 b7 ff ff ff       	callq  40ac80 <__sprintf_chk@plt+0x83f0>
  40acc9:	49 89 06             	mov    %rax,(%r14)
  40accc:	48 8b 30             	mov    (%rax),%rsi
  40accf:	48 89 c3             	mov    %rax,%rbx
  40acd2:	48 85 f6             	test   %rsi,%rsi
  40acd5:	74 78                	je     40ad4f <__sprintf_chk@plt+0x84bf>
  40acd7:	49 39 f4             	cmp    %rsi,%r12
  40acda:	74 0d                	je     40ace9 <__sprintf_chk@plt+0x8459>
  40acdc:	4c 89 e7             	mov    %r12,%rdi
  40acdf:	ff 55 38             	callq  *0x38(%rbp)
  40ace2:	84 c0                	test   %al,%al
  40ace4:	74 60                	je     40ad46 <__sprintf_chk@plt+0x84b6>
  40ace6:	48 8b 33             	mov    (%rbx),%rsi
  40ace9:	45 84 ed             	test   %r13b,%r13b
  40acec:	74 2e                	je     40ad1c <__sprintf_chk@plt+0x848c>
  40acee:	48 8b 43 08          	mov    0x8(%rbx),%rax
  40acf2:	48 85 c0             	test   %rax,%rax
  40acf5:	0f 84 9d 00 00 00    	je     40ad98 <__sprintf_chk@plt+0x8508>
  40acfb:	4c 8b 08             	mov    (%rax),%r9
  40acfe:	4c 8b 50 08          	mov    0x8(%rax),%r10
  40ad02:	4c 89 0b             	mov    %r9,(%rbx)
  40ad05:	4c 89 53 08          	mov    %r10,0x8(%rbx)
  40ad09:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  40ad10:	48 8b 4d 48          	mov    0x48(%rbp),%rcx
  40ad14:	48 89 48 08          	mov    %rcx,0x8(%rax)
  40ad18:	48 89 45 48          	mov    %rax,0x48(%rbp)
  40ad1c:	5b                   	pop    %rbx
  40ad1d:	5d                   	pop    %rbp
  40ad1e:	41 5c                	pop    %r12
  40ad20:	41 5d                	pop    %r13
  40ad22:	48 89 f0             	mov    %rsi,%rax
  40ad25:	41 5e                	pop    %r14
  40ad27:	c3                   	retq   
  40ad28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40ad2f:	00 
  40ad30:	48 8b 30             	mov    (%rax),%rsi
  40ad33:	4c 39 e6             	cmp    %r12,%rsi
  40ad36:	74 2f                	je     40ad67 <__sprintf_chk@plt+0x84d7>
  40ad38:	4c 89 e7             	mov    %r12,%rdi
  40ad3b:	ff 55 38             	callq  *0x38(%rbp)
  40ad3e:	84 c0                	test   %al,%al
  40ad40:	75 1e                	jne    40ad60 <__sprintf_chk@plt+0x84d0>
  40ad42:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40ad46:	48 8b 43 08          	mov    0x8(%rbx),%rax
  40ad4a:	48 85 c0             	test   %rax,%rax
  40ad4d:	75 e1                	jne    40ad30 <__sprintf_chk@plt+0x84a0>
  40ad4f:	5b                   	pop    %rbx
  40ad50:	5d                   	pop    %rbp
  40ad51:	41 5c                	pop    %r12
  40ad53:	31 f6                	xor    %esi,%esi
  40ad55:	41 5d                	pop    %r13
  40ad57:	48 89 f0             	mov    %rsi,%rax
  40ad5a:	41 5e                	pop    %r14
  40ad5c:	c3                   	retq   
  40ad5d:	0f 1f 00             	nopl   (%rax)
  40ad60:	48 8b 43 08          	mov    0x8(%rbx),%rax
  40ad64:	48 8b 30             	mov    (%rax),%rsi
  40ad67:	45 84 ed             	test   %r13b,%r13b
  40ad6a:	74 b0                	je     40ad1c <__sprintf_chk@plt+0x848c>
  40ad6c:	48 8b 48 08          	mov    0x8(%rax),%rcx
  40ad70:	48 89 4b 08          	mov    %rcx,0x8(%rbx)
  40ad74:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  40ad7b:	48 8b 4d 48          	mov    0x48(%rbp),%rcx
  40ad7f:	48 89 48 08          	mov    %rcx,0x8(%rax)
  40ad83:	48 89 45 48          	mov    %rax,0x48(%rbp)
  40ad87:	48 89 f0             	mov    %rsi,%rax
  40ad8a:	5b                   	pop    %rbx
  40ad8b:	5d                   	pop    %rbp
  40ad8c:	41 5c                	pop    %r12
  40ad8e:	41 5d                	pop    %r13
  40ad90:	41 5e                	pop    %r14
  40ad92:	c3                   	retq   
  40ad93:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ad98:	48 c7 03 00 00 00 00 	movq   $0x0,(%rbx)
  40ad9f:	e9 78 ff ff ff       	jmpq   40ad1c <__sprintf_chk@plt+0x848c>
  40ada4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40adab:	00 00 00 00 00 
  40adb0:	48 8b 07             	mov    (%rdi),%rax
  40adb3:	48 3d e0 5e 41 00    	cmp    $0x415ee0,%rax
  40adb9:	74 75                	je     40ae30 <__sprintf_chk@plt+0x85a0>
  40adbb:	f3 0f 10 40 08       	movss  0x8(%rax),%xmm0
  40adc0:	0f 2e 05 2d b1 00 00 	ucomiss 0xb12d(%rip),%xmm0        # 415ef4 <_fini@@Base+0x3ff8>
  40adc7:	76 57                	jbe    40ae20 <__sprintf_chk@plt+0x8590>
  40adc9:	f3 0f 10 0d 27 b1 00 	movss  0xb127(%rip),%xmm1        # 415ef8 <_fini@@Base+0x3ffc>
  40add0:	00 
  40add1:	0f 2e c8             	ucomiss %xmm0,%xmm1
  40add4:	76 4a                	jbe    40ae20 <__sprintf_chk@plt+0x8590>
  40add6:	f3 0f 10 48 0c       	movss  0xc(%rax),%xmm1
  40addb:	0f 2e 0d 1a b1 00 00 	ucomiss 0xb11a(%rip),%xmm1        # 415efc <_fini@@Base+0x4000>
  40ade2:	76 3c                	jbe    40ae20 <__sprintf_chk@plt+0x8590>
  40ade4:	f3 0f 10 08          	movss  (%rax),%xmm1
  40ade8:	0f 2e 0d 11 b1 00 00 	ucomiss 0xb111(%rip),%xmm1        # 415f00 <_fini@@Base+0x4004>
  40adef:	72 2f                	jb     40ae20 <__sprintf_chk@plt+0x8590>
  40adf1:	f3 0f 58 0d fb b0 00 	addss  0xb0fb(%rip),%xmm1        # 415ef4 <_fini@@Base+0x3ff8>
  40adf8:	00 
  40adf9:	f3 0f 10 50 04       	movss  0x4(%rax),%xmm2
  40adfe:	0f 2e d1             	ucomiss %xmm1,%xmm2
  40ae01:	76 1d                	jbe    40ae20 <__sprintf_chk@plt+0x8590>
  40ae03:	f3 0f 10 1d f9 b0 00 	movss  0xb0f9(%rip),%xmm3        # 415f04 <_fini@@Base+0x4008>
  40ae0a:	00 
  40ae0b:	0f 2e da             	ucomiss %xmm2,%xmm3
  40ae0e:	72 10                	jb     40ae20 <__sprintf_chk@plt+0x8590>
  40ae10:	0f 2e c1             	ucomiss %xmm1,%xmm0
  40ae13:	b8 01 00 00 00       	mov    $0x1,%eax
  40ae18:	77 1b                	ja     40ae35 <__sprintf_chk@plt+0x85a5>
  40ae1a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ae20:	48 c7 07 e0 5e 41 00 	movq   $0x415ee0,(%rdi)
  40ae27:	31 c0                	xor    %eax,%eax
  40ae29:	c3                   	retq   
  40ae2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ae30:	b8 01 00 00 00       	mov    $0x1,%eax
  40ae35:	f3 c3                	repz retq 
  40ae37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40ae3e:	00 00 
  40ae40:	41 57                	push   %r15
  40ae42:	41 89 d7             	mov    %edx,%r15d
  40ae45:	41 56                	push   %r14
  40ae47:	49 89 f6             	mov    %rsi,%r14
  40ae4a:	41 55                	push   %r13
  40ae4c:	41 54                	push   %r12
  40ae4e:	49 89 fc             	mov    %rdi,%r12
  40ae51:	55                   	push   %rbp
  40ae52:	53                   	push   %rbx
  40ae53:	48 83 ec 08          	sub    $0x8,%rsp
  40ae57:	4c 8b 2e             	mov    (%rsi),%r13
  40ae5a:	4c 3b 6e 08          	cmp    0x8(%rsi),%r13
  40ae5e:	0f 83 8b 00 00 00    	jae    40aeef <__sprintf_chk@plt+0x865f>
  40ae64:	0f 1f 40 00          	nopl   0x0(%rax)
  40ae68:	49 8b 6d 00          	mov    0x0(%r13),%rbp
  40ae6c:	48 85 ed             	test   %rbp,%rbp
  40ae6f:	74 70                	je     40aee1 <__sprintf_chk@plt+0x8651>
  40ae71:	49 8b 5d 08          	mov    0x8(%r13),%rbx
  40ae75:	48 85 db             	test   %rbx,%rbx
  40ae78:	75 1a                	jne    40ae94 <__sprintf_chk@plt+0x8604>
  40ae7a:	eb 58                	jmp    40aed4 <__sprintf_chk@plt+0x8644>
  40ae7c:	0f 1f 40 00          	nopl   0x0(%rax)
  40ae80:	48 8b 48 08          	mov    0x8(%rax),%rcx
  40ae84:	48 85 d2             	test   %rdx,%rdx
  40ae87:	48 89 4b 08          	mov    %rcx,0x8(%rbx)
  40ae8b:	48 89 58 08          	mov    %rbx,0x8(%rax)
  40ae8f:	74 3f                	je     40aed0 <__sprintf_chk@plt+0x8640>
  40ae91:	48 89 d3             	mov    %rdx,%rbx
  40ae94:	48 8b 2b             	mov    (%rbx),%rbp
  40ae97:	4c 89 e7             	mov    %r12,%rdi
  40ae9a:	48 89 ee             	mov    %rbp,%rsi
  40ae9d:	e8 de fd ff ff       	callq  40ac80 <__sprintf_chk@plt+0x83f0>
  40aea2:	48 83 38 00          	cmpq   $0x0,(%rax)
  40aea6:	48 8b 53 08          	mov    0x8(%rbx),%rdx
  40aeaa:	75 d4                	jne    40ae80 <__sprintf_chk@plt+0x85f0>
  40aeac:	48 89 28             	mov    %rbp,(%rax)
  40aeaf:	49 83 44 24 18 01    	addq   $0x1,0x18(%r12)
  40aeb5:	48 85 d2             	test   %rdx,%rdx
  40aeb8:	48 c7 03 00 00 00 00 	movq   $0x0,(%rbx)
  40aebf:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  40aec4:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40aec8:	49 89 5c 24 48       	mov    %rbx,0x48(%r12)
  40aecd:	75 c2                	jne    40ae91 <__sprintf_chk@plt+0x8601>
  40aecf:	90                   	nop
  40aed0:	49 8b 6d 00          	mov    0x0(%r13),%rbp
  40aed4:	45 84 ff             	test   %r15b,%r15b
  40aed7:	49 c7 45 08 00 00 00 	movq   $0x0,0x8(%r13)
  40aede:	00 
  40aedf:	74 27                	je     40af08 <__sprintf_chk@plt+0x8678>
  40aee1:	49 83 c5 10          	add    $0x10,%r13
  40aee5:	4d 39 6e 08          	cmp    %r13,0x8(%r14)
  40aee9:	0f 87 79 ff ff ff    	ja     40ae68 <__sprintf_chk@plt+0x85d8>
  40aeef:	48 83 c4 08          	add    $0x8,%rsp
  40aef3:	b8 01 00 00 00       	mov    $0x1,%eax
  40aef8:	5b                   	pop    %rbx
  40aef9:	5d                   	pop    %rbp
  40aefa:	41 5c                	pop    %r12
  40aefc:	41 5d                	pop    %r13
  40aefe:	41 5e                	pop    %r14
  40af00:	41 5f                	pop    %r15
  40af02:	c3                   	retq   
  40af03:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40af08:	48 89 ee             	mov    %rbp,%rsi
  40af0b:	4c 89 e7             	mov    %r12,%rdi
  40af0e:	e8 6d fd ff ff       	callq  40ac80 <__sprintf_chk@plt+0x83f0>
  40af13:	48 83 38 00          	cmpq   $0x0,(%rax)
  40af17:	48 89 c3             	mov    %rax,%rbx
  40af1a:	74 3f                	je     40af5b <__sprintf_chk@plt+0x86cb>
  40af1c:	49 8b 44 24 48       	mov    0x48(%r12),%rax
  40af21:	48 85 c0             	test   %rax,%rax
  40af24:	74 40                	je     40af66 <__sprintf_chk@plt+0x86d6>
  40af26:	48 8b 50 08          	mov    0x8(%rax),%rdx
  40af2a:	49 89 54 24 48       	mov    %rdx,0x48(%r12)
  40af2f:	48 8b 53 08          	mov    0x8(%rbx),%rdx
  40af33:	48 89 28             	mov    %rbp,(%rax)
  40af36:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40af3a:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40af3e:	49 c7 45 00 00 00 00 	movq   $0x0,0x0(%r13)
  40af45:	00 
  40af46:	49 83 6e 18 01       	subq   $0x1,0x18(%r14)
  40af4b:	49 83 c5 10          	add    $0x10,%r13
  40af4f:	4d 39 6e 08          	cmp    %r13,0x8(%r14)
  40af53:	0f 87 0f ff ff ff    	ja     40ae68 <__sprintf_chk@plt+0x85d8>
  40af59:	eb 94                	jmp    40aeef <__sprintf_chk@plt+0x865f>
  40af5b:	48 89 28             	mov    %rbp,(%rax)
  40af5e:	49 83 44 24 18 01    	addq   $0x1,0x18(%r12)
  40af64:	eb d8                	jmp    40af3e <__sprintf_chk@plt+0x86ae>
  40af66:	bf 10 00 00 00       	mov    $0x10,%edi
  40af6b:	e8 d0 76 ff ff       	callq  402640 <malloc@plt>
  40af70:	48 85 c0             	test   %rax,%rax
  40af73:	75 ba                	jne    40af2f <__sprintf_chk@plt+0x869f>
  40af75:	48 83 c4 08          	add    $0x8,%rsp
  40af79:	31 c0                	xor    %eax,%eax
  40af7b:	5b                   	pop    %rbx
  40af7c:	5d                   	pop    %rbp
  40af7d:	41 5c                	pop    %r12
  40af7f:	41 5d                	pop    %r13
  40af81:	41 5e                	pop    %r14
  40af83:	41 5f                	pop    %r15
  40af85:	c3                   	retq   
  40af86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40af8d:	00 00 00 
  40af90:	48 8b 47 10          	mov    0x10(%rdi),%rax
  40af94:	c3                   	retq   
  40af95:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40af9c:	00 00 00 00 
  40afa0:	48 8b 47 18          	mov    0x18(%rdi),%rax
  40afa4:	c3                   	retq   
  40afa5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40afac:	00 00 00 00 
  40afb0:	48 8b 47 20          	mov    0x20(%rdi),%rax
  40afb4:	c3                   	retq   
  40afb5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40afbc:	00 00 00 00 
  40afc0:	48 8b 37             	mov    (%rdi),%rsi
  40afc3:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  40afc7:	31 c0                	xor    %eax,%eax
  40afc9:	48 39 fe             	cmp    %rdi,%rsi
  40afcc:	73 39                	jae    40b007 <__sprintf_chk@plt+0x8777>
  40afce:	66 90                	xchg   %ax,%ax
  40afd0:	48 83 3e 00          	cmpq   $0x0,(%rsi)
  40afd4:	74 26                	je     40affc <__sprintf_chk@plt+0x876c>
  40afd6:	48 8b 56 08          	mov    0x8(%rsi),%rdx
  40afda:	b9 01 00 00 00       	mov    $0x1,%ecx
  40afdf:	48 85 d2             	test   %rdx,%rdx
  40afe2:	74 11                	je     40aff5 <__sprintf_chk@plt+0x8765>
  40afe4:	0f 1f 40 00          	nopl   0x0(%rax)
  40afe8:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40afec:	48 83 c1 01          	add    $0x1,%rcx
  40aff0:	48 85 d2             	test   %rdx,%rdx
  40aff3:	75 f3                	jne    40afe8 <__sprintf_chk@plt+0x8758>
  40aff5:	48 39 c8             	cmp    %rcx,%rax
  40aff8:	48 0f 42 c1          	cmovb  %rcx,%rax
  40affc:	48 83 c6 10          	add    $0x10,%rsi
  40b000:	48 39 fe             	cmp    %rdi,%rsi
  40b003:	72 cb                	jb     40afd0 <__sprintf_chk@plt+0x8740>
  40b005:	f3 c3                	repz retq 
  40b007:	f3 c3                	repz retq 
  40b009:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40b010:	48 8b 0f             	mov    (%rdi),%rcx
  40b013:	4c 8b 47 08          	mov    0x8(%rdi),%r8
  40b017:	31 d2                	xor    %edx,%edx
  40b019:	31 f6                	xor    %esi,%esi
  40b01b:	4c 39 c1             	cmp    %r8,%rcx
  40b01e:	73 36                	jae    40b056 <__sprintf_chk@plt+0x87c6>
  40b020:	48 83 39 00          	cmpq   $0x0,(%rcx)
  40b024:	74 27                	je     40b04d <__sprintf_chk@plt+0x87bd>
  40b026:	48 8b 41 08          	mov    0x8(%rcx),%rax
  40b02a:	48 83 c6 01          	add    $0x1,%rsi
  40b02e:	48 83 c2 01          	add    $0x1,%rdx
  40b032:	48 85 c0             	test   %rax,%rax
  40b035:	74 16                	je     40b04d <__sprintf_chk@plt+0x87bd>
  40b037:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40b03e:	00 00 
  40b040:	48 8b 40 08          	mov    0x8(%rax),%rax
  40b044:	48 83 c2 01          	add    $0x1,%rdx
  40b048:	48 85 c0             	test   %rax,%rax
  40b04b:	75 f3                	jne    40b040 <__sprintf_chk@plt+0x87b0>
  40b04d:	48 83 c1 10          	add    $0x10,%rcx
  40b051:	4c 39 c1             	cmp    %r8,%rcx
  40b054:	72 ca                	jb     40b020 <__sprintf_chk@plt+0x8790>
  40b056:	31 c0                	xor    %eax,%eax
  40b058:	48 39 77 18          	cmp    %rsi,0x18(%rdi)
  40b05c:	74 02                	je     40b060 <__sprintf_chk@plt+0x87d0>
  40b05e:	f3 c3                	repz retq 
  40b060:	48 39 57 20          	cmp    %rdx,0x20(%rdi)
  40b064:	0f 94 c0             	sete   %al
  40b067:	c3                   	retq   
  40b068:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40b06f:	00 
  40b070:	41 55                	push   %r13
  40b072:	41 54                	push   %r12
  40b074:	55                   	push   %rbp
  40b075:	48 89 f5             	mov    %rsi,%rbp
  40b078:	53                   	push   %rbx
  40b079:	31 db                	xor    %ebx,%ebx
  40b07b:	48 83 ec 08          	sub    $0x8,%rsp
  40b07f:	4c 8b 07             	mov    (%rdi),%r8
  40b082:	48 8b 77 08          	mov    0x8(%rdi),%rsi
  40b086:	48 8b 4f 20          	mov    0x20(%rdi),%rcx
  40b08a:	4c 8b 67 10          	mov    0x10(%rdi),%r12
  40b08e:	4c 8b 6f 18          	mov    0x18(%rdi),%r13
  40b092:	49 39 f0             	cmp    %rsi,%r8
  40b095:	73 3e                	jae    40b0d5 <__sprintf_chk@plt+0x8845>
  40b097:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40b09e:	00 00 
  40b0a0:	49 83 38 00          	cmpq   $0x0,(%r8)
  40b0a4:	74 26                	je     40b0cc <__sprintf_chk@plt+0x883c>
  40b0a6:	49 8b 40 08          	mov    0x8(%r8),%rax
  40b0aa:	ba 01 00 00 00       	mov    $0x1,%edx
  40b0af:	48 85 c0             	test   %rax,%rax
  40b0b2:	74 11                	je     40b0c5 <__sprintf_chk@plt+0x8835>
  40b0b4:	0f 1f 40 00          	nopl   0x0(%rax)
  40b0b8:	48 8b 40 08          	mov    0x8(%rax),%rax
  40b0bc:	48 83 c2 01          	add    $0x1,%rdx
  40b0c0:	48 85 c0             	test   %rax,%rax
  40b0c3:	75 f3                	jne    40b0b8 <__sprintf_chk@plt+0x8828>
  40b0c5:	48 39 d3             	cmp    %rdx,%rbx
  40b0c8:	48 0f 42 da          	cmovb  %rdx,%rbx
  40b0cc:	49 83 c0 10          	add    $0x10,%r8
  40b0d0:	49 39 f0             	cmp    %rsi,%r8
  40b0d3:	72 cb                	jb     40b0a0 <__sprintf_chk@plt+0x8810>
  40b0d5:	ba 70 5e 41 00       	mov    $0x415e70,%edx
  40b0da:	be 01 00 00 00       	mov    $0x1,%esi
  40b0df:	48 89 ef             	mov    %rbp,%rdi
  40b0e2:	31 c0                	xor    %eax,%eax
  40b0e4:	e8 27 77 ff ff       	callq  402810 <__fprintf_chk@plt>
  40b0e9:	31 c0                	xor    %eax,%eax
  40b0eb:	4c 89 e1             	mov    %r12,%rcx
  40b0ee:	ba 88 5e 41 00       	mov    $0x415e88,%edx
  40b0f3:	be 01 00 00 00       	mov    $0x1,%esi
  40b0f8:	48 89 ef             	mov    %rbp,%rdi
  40b0fb:	e8 10 77 ff ff       	callq  402810 <__fprintf_chk@plt>
  40b100:	4d 85 ed             	test   %r13,%r13
  40b103:	78 56                	js     40b15b <__sprintf_chk@plt+0x88cb>
  40b105:	f2 49 0f 2a c5       	cvtsi2sd %r13,%xmm0
  40b10a:	4d 85 e4             	test   %r12,%r12
  40b10d:	f2 0f 59 05 fb ad 00 	mulsd  0xadfb(%rip),%xmm0        # 415f10 <_fini@@Base+0x4014>
  40b114:	00 
  40b115:	78 69                	js     40b180 <__sprintf_chk@plt+0x88f0>
  40b117:	f2 49 0f 2a cc       	cvtsi2sd %r12,%xmm1
  40b11c:	f2 0f 5e c1          	divsd  %xmm1,%xmm0
  40b120:	4c 89 e9             	mov    %r13,%rcx
  40b123:	48 89 ef             	mov    %rbp,%rdi
  40b126:	ba b8 5e 41 00       	mov    $0x415eb8,%edx
  40b12b:	be 01 00 00 00       	mov    $0x1,%esi
  40b130:	b8 01 00 00 00       	mov    $0x1,%eax
  40b135:	e8 d6 76 ff ff       	callq  402810 <__fprintf_chk@plt>
  40b13a:	48 83 c4 08          	add    $0x8,%rsp
  40b13e:	48 89 d9             	mov    %rbx,%rcx
  40b141:	48 89 ef             	mov    %rbp,%rdi
  40b144:	5b                   	pop    %rbx
  40b145:	5d                   	pop    %rbp
  40b146:	41 5c                	pop    %r12
  40b148:	41 5d                	pop    %r13
  40b14a:	ba a0 5e 41 00       	mov    $0x415ea0,%edx
  40b14f:	be 01 00 00 00       	mov    $0x1,%esi
  40b154:	31 c0                	xor    %eax,%eax
  40b156:	e9 b5 76 ff ff       	jmpq   402810 <__fprintf_chk@plt>
  40b15b:	4c 89 e8             	mov    %r13,%rax
  40b15e:	4c 89 ea             	mov    %r13,%rdx
  40b161:	48 d1 e8             	shr    %rax
  40b164:	83 e2 01             	and    $0x1,%edx
  40b167:	48 09 d0             	or     %rdx,%rax
  40b16a:	4d 85 e4             	test   %r12,%r12
  40b16d:	f2 48 0f 2a c0       	cvtsi2sd %rax,%xmm0
  40b172:	f2 0f 58 c0          	addsd  %xmm0,%xmm0
  40b176:	f2 0f 59 05 92 ad 00 	mulsd  0xad92(%rip),%xmm0        # 415f10 <_fini@@Base+0x4014>
  40b17d:	00 
  40b17e:	79 97                	jns    40b117 <__sprintf_chk@plt+0x8887>
  40b180:	4c 89 e0             	mov    %r12,%rax
  40b183:	41 83 e4 01          	and    $0x1,%r12d
  40b187:	48 d1 e8             	shr    %rax
  40b18a:	4c 09 e0             	or     %r12,%rax
  40b18d:	f2 48 0f 2a c8       	cvtsi2sd %rax,%xmm1
  40b192:	f2 0f 58 c9          	addsd  %xmm1,%xmm1
  40b196:	eb 84                	jmp    40b11c <__sprintf_chk@plt+0x888c>
  40b198:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40b19f:	00 
  40b1a0:	41 54                	push   %r12
  40b1a2:	49 89 fc             	mov    %rdi,%r12
  40b1a5:	55                   	push   %rbp
  40b1a6:	48 89 f5             	mov    %rsi,%rbp
  40b1a9:	53                   	push   %rbx
  40b1aa:	e8 d1 fa ff ff       	callq  40ac80 <__sprintf_chk@plt+0x83f0>
  40b1af:	48 89 c3             	mov    %rax,%rbx
  40b1b2:	48 8b 00             	mov    (%rax),%rax
  40b1b5:	48 85 c0             	test   %rax,%rax
  40b1b8:	74 23                	je     40b1dd <__sprintf_chk@plt+0x894d>
  40b1ba:	48 89 c6             	mov    %rax,%rsi
  40b1bd:	eb 04                	jmp    40b1c3 <__sprintf_chk@plt+0x8933>
  40b1bf:	90                   	nop
  40b1c0:	48 8b 33             	mov    (%rbx),%rsi
  40b1c3:	48 39 f5             	cmp    %rsi,%rbp
  40b1c6:	74 23                	je     40b1eb <__sprintf_chk@plt+0x895b>
  40b1c8:	48 89 ef             	mov    %rbp,%rdi
  40b1cb:	41 ff 54 24 38       	callq  *0x38(%r12)
  40b1d0:	84 c0                	test   %al,%al
  40b1d2:	75 14                	jne    40b1e8 <__sprintf_chk@plt+0x8958>
  40b1d4:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40b1d8:	48 85 db             	test   %rbx,%rbx
  40b1db:	75 e3                	jne    40b1c0 <__sprintf_chk@plt+0x8930>
  40b1dd:	5b                   	pop    %rbx
  40b1de:	5d                   	pop    %rbp
  40b1df:	31 c0                	xor    %eax,%eax
  40b1e1:	41 5c                	pop    %r12
  40b1e3:	c3                   	retq   
  40b1e4:	0f 1f 40 00          	nopl   0x0(%rax)
  40b1e8:	48 8b 33             	mov    (%rbx),%rsi
  40b1eb:	5b                   	pop    %rbx
  40b1ec:	5d                   	pop    %rbp
  40b1ed:	48 89 f0             	mov    %rsi,%rax
  40b1f0:	41 5c                	pop    %r12
  40b1f2:	c3                   	retq   
  40b1f3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b1fa:	84 00 00 00 00 00 
  40b200:	48 83 ec 08          	sub    $0x8,%rsp
  40b204:	48 83 7f 20 00       	cmpq   $0x0,0x20(%rdi)
  40b209:	74 2b                	je     40b236 <__sprintf_chk@plt+0x89a6>
  40b20b:	48 8b 17             	mov    (%rdi),%rdx
  40b20e:	48 8b 4f 08          	mov    0x8(%rdi),%rcx
  40b212:	48 39 ca             	cmp    %rcx,%rdx
  40b215:	72 12                	jb     40b229 <__sprintf_chk@plt+0x8999>
  40b217:	eb 24                	jmp    40b23d <__sprintf_chk@plt+0x89ad>
  40b219:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40b220:	48 83 c2 10          	add    $0x10,%rdx
  40b224:	48 39 ca             	cmp    %rcx,%rdx
  40b227:	73 14                	jae    40b23d <__sprintf_chk@plt+0x89ad>
  40b229:	48 8b 02             	mov    (%rdx),%rax
  40b22c:	48 85 c0             	test   %rax,%rax
  40b22f:	74 ef                	je     40b220 <__sprintf_chk@plt+0x8990>
  40b231:	48 83 c4 08          	add    $0x8,%rsp
  40b235:	c3                   	retq   
  40b236:	31 c0                	xor    %eax,%eax
  40b238:	48 83 c4 08          	add    $0x8,%rsp
  40b23c:	c3                   	retq   
  40b23d:	e8 de 6f ff ff       	callq  402220 <abort@plt>
  40b242:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b249:	1f 84 00 00 00 00 00 
  40b250:	55                   	push   %rbp
  40b251:	48 89 fd             	mov    %rdi,%rbp
  40b254:	53                   	push   %rbx
  40b255:	48 89 f3             	mov    %rsi,%rbx
  40b258:	48 83 ec 08          	sub    $0x8,%rsp
  40b25c:	e8 1f fa ff ff       	callq  40ac80 <__sprintf_chk@plt+0x83f0>
  40b261:	48 89 c1             	mov    %rax,%rcx
  40b264:	48 89 c2             	mov    %rax,%rdx
  40b267:	eb 10                	jmp    40b279 <__sprintf_chk@plt+0x89e9>
  40b269:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40b270:	48 8b 52 08          	mov    0x8(%rdx),%rdx
  40b274:	48 85 d2             	test   %rdx,%rdx
  40b277:	74 0e                	je     40b287 <__sprintf_chk@plt+0x89f7>
  40b279:	48 39 1a             	cmp    %rbx,(%rdx)
  40b27c:	75 f2                	jne    40b270 <__sprintf_chk@plt+0x89e0>
  40b27e:	48 8b 42 08          	mov    0x8(%rdx),%rax
  40b282:	48 85 c0             	test   %rax,%rax
  40b285:	75 23                	jne    40b2aa <__sprintf_chk@plt+0x8a1a>
  40b287:	48 8b 55 08          	mov    0x8(%rbp),%rdx
  40b28b:	eb 0b                	jmp    40b298 <__sprintf_chk@plt+0x8a08>
  40b28d:	0f 1f 00             	nopl   (%rax)
  40b290:	48 8b 01             	mov    (%rcx),%rax
  40b293:	48 85 c0             	test   %rax,%rax
  40b296:	75 0b                	jne    40b2a3 <__sprintf_chk@plt+0x8a13>
  40b298:	48 83 c1 10          	add    $0x10,%rcx
  40b29c:	48 39 d1             	cmp    %rdx,%rcx
  40b29f:	72 ef                	jb     40b290 <__sprintf_chk@plt+0x8a00>
  40b2a1:	31 c0                	xor    %eax,%eax
  40b2a3:	48 83 c4 08          	add    $0x8,%rsp
  40b2a7:	5b                   	pop    %rbx
  40b2a8:	5d                   	pop    %rbp
  40b2a9:	c3                   	retq   
  40b2aa:	48 8b 00             	mov    (%rax),%rax
  40b2ad:	48 83 c4 08          	add    $0x8,%rsp
  40b2b1:	5b                   	pop    %rbx
  40b2b2:	5d                   	pop    %rbp
  40b2b3:	c3                   	retq   
  40b2b4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b2bb:	00 00 00 00 00 
  40b2c0:	4c 8b 0f             	mov    (%rdi),%r9
  40b2c3:	31 c0                	xor    %eax,%eax
  40b2c5:	4c 39 4f 08          	cmp    %r9,0x8(%rdi)
  40b2c9:	76 57                	jbe    40b322 <__sprintf_chk@plt+0x8a92>
  40b2cb:	49 8b 09             	mov    (%r9),%rcx
  40b2ce:	48 85 c9             	test   %rcx,%rcx
  40b2d1:	74 37                	je     40b30a <__sprintf_chk@plt+0x8a7a>
  40b2d3:	48 39 c2             	cmp    %rax,%rdx
  40b2d6:	76 48                	jbe    40b320 <__sprintf_chk@plt+0x8a90>
  40b2d8:	48 89 0c c6          	mov    %rcx,(%rsi,%rax,8)
  40b2dc:	49 8b 49 08          	mov    0x8(%r9),%rcx
  40b2e0:	4c 8d 40 01          	lea    0x1(%rax),%r8
  40b2e4:	4c 89 c0             	mov    %r8,%rax
  40b2e7:	48 85 c9             	test   %rcx,%rcx
  40b2ea:	74 1e                	je     40b30a <__sprintf_chk@plt+0x8a7a>
  40b2ec:	0f 1f 40 00          	nopl   0x0(%rax)
  40b2f0:	48 39 d0             	cmp    %rdx,%rax
  40b2f3:	74 2b                	je     40b320 <__sprintf_chk@plt+0x8a90>
  40b2f5:	4c 8b 01             	mov    (%rcx),%r8
  40b2f8:	48 83 c0 01          	add    $0x1,%rax
  40b2fc:	4c 89 44 c6 f8       	mov    %r8,-0x8(%rsi,%rax,8)
  40b301:	48 8b 49 08          	mov    0x8(%rcx),%rcx
  40b305:	48 85 c9             	test   %rcx,%rcx
  40b308:	75 e6                	jne    40b2f0 <__sprintf_chk@plt+0x8a60>
  40b30a:	49 83 c1 10          	add    $0x10,%r9
  40b30e:	4c 39 4f 08          	cmp    %r9,0x8(%rdi)
  40b312:	77 b7                	ja     40b2cb <__sprintf_chk@plt+0x8a3b>
  40b314:	f3 c3                	repz retq 
  40b316:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40b31d:	00 00 00 
  40b320:	f3 c3                	repz retq 
  40b322:	f3 c3                	repz retq 
  40b324:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b32b:	00 00 00 00 00 
  40b330:	41 57                	push   %r15
  40b332:	49 89 ff             	mov    %rdi,%r15
  40b335:	41 56                	push   %r14
  40b337:	41 55                	push   %r13
  40b339:	41 54                	push   %r12
  40b33b:	55                   	push   %rbp
  40b33c:	53                   	push   %rbx
  40b33d:	48 83 ec 08          	sub    $0x8,%rsp
  40b341:	4c 8b 37             	mov    (%rdi),%r14
  40b344:	4c 39 77 08          	cmp    %r14,0x8(%rdi)
  40b348:	76 50                	jbe    40b39a <__sprintf_chk@plt+0x8b0a>
  40b34a:	49 89 f4             	mov    %rsi,%r12
  40b34d:	49 89 d5             	mov    %rdx,%r13
  40b350:	31 ed                	xor    %ebp,%ebp
  40b352:	49 8b 3e             	mov    (%r14),%rdi
  40b355:	48 85 ff             	test   %rdi,%rdi
  40b358:	74 20                	je     40b37a <__sprintf_chk@plt+0x8aea>
  40b35a:	4c 89 f3             	mov    %r14,%rbx
  40b35d:	eb 04                	jmp    40b363 <__sprintf_chk@plt+0x8ad3>
  40b35f:	90                   	nop
  40b360:	48 8b 3b             	mov    (%rbx),%rdi
  40b363:	4c 89 ee             	mov    %r13,%rsi
  40b366:	41 ff d4             	callq  *%r12
  40b369:	84 c0                	test   %al,%al
  40b36b:	74 1b                	je     40b388 <__sprintf_chk@plt+0x8af8>
  40b36d:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40b371:	48 83 c5 01          	add    $0x1,%rbp
  40b375:	48 85 db             	test   %rbx,%rbx
  40b378:	75 e6                	jne    40b360 <__sprintf_chk@plt+0x8ad0>
  40b37a:	49 83 c6 10          	add    $0x10,%r14
  40b37e:	4d 39 77 08          	cmp    %r14,0x8(%r15)
  40b382:	77 ce                	ja     40b352 <__sprintf_chk@plt+0x8ac2>
  40b384:	0f 1f 40 00          	nopl   0x0(%rax)
  40b388:	48 83 c4 08          	add    $0x8,%rsp
  40b38c:	48 89 e8             	mov    %rbp,%rax
  40b38f:	5b                   	pop    %rbx
  40b390:	5d                   	pop    %rbp
  40b391:	41 5c                	pop    %r12
  40b393:	41 5d                	pop    %r13
  40b395:	41 5e                	pop    %r14
  40b397:	41 5f                	pop    %r15
  40b399:	c3                   	retq   
  40b39a:	31 ed                	xor    %ebp,%ebp
  40b39c:	eb ea                	jmp    40b388 <__sprintf_chk@plt+0x8af8>
  40b39e:	66 90                	xchg   %ax,%ax
  40b3a0:	44 0f b6 07          	movzbl (%rdi),%r8d
  40b3a4:	31 d2                	xor    %edx,%edx
  40b3a6:	45 84 c0             	test   %r8b,%r8b
  40b3a9:	74 25                	je     40b3d0 <__sprintf_chk@plt+0x8b40>
  40b3ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40b3b0:	48 89 d1             	mov    %rdx,%rcx
  40b3b3:	48 83 c7 01          	add    $0x1,%rdi
  40b3b7:	48 c1 e1 05          	shl    $0x5,%rcx
  40b3bb:	48 29 d1             	sub    %rdx,%rcx
  40b3be:	31 d2                	xor    %edx,%edx
  40b3c0:	49 8d 04 08          	lea    (%r8,%rcx,1),%rax
  40b3c4:	44 0f b6 07          	movzbl (%rdi),%r8d
  40b3c8:	48 f7 f6             	div    %rsi
  40b3cb:	45 84 c0             	test   %r8b,%r8b
  40b3ce:	75 e0                	jne    40b3b0 <__sprintf_chk@plt+0x8b20>
  40b3d0:	48 89 d0             	mov    %rdx,%rax
  40b3d3:	c3                   	retq   
  40b3d4:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b3db:	00 00 00 00 00 
  40b3e0:	c7 07 00 00 00 00    	movl   $0x0,(%rdi)
  40b3e6:	c7 47 04 00 00 80 3f 	movl   $0x3f800000,0x4(%rdi)
  40b3ed:	c7 47 08 cd cc 4c 3f 	movl   $0x3f4ccccd,0x8(%rdi)
  40b3f4:	c7 47 0c f4 fd b4 3f 	movl   $0x3fb4fdf4,0xc(%rdi)
  40b3fb:	c6 47 10 00          	movb   $0x0,0x10(%rdi)
  40b3ff:	c3                   	retq   
  40b400:	41 57                	push   %r15
  40b402:	b8 60 ac 40 00       	mov    $0x40ac60,%eax
  40b407:	49 89 ff             	mov    %rdi,%r15
  40b40a:	bf 50 00 00 00       	mov    $0x50,%edi
  40b40f:	41 56                	push   %r14
  40b411:	4d 89 c6             	mov    %r8,%r14
  40b414:	41 55                	push   %r13
  40b416:	49 89 d5             	mov    %rdx,%r13
  40b419:	41 54                	push   %r12
  40b41b:	49 89 cc             	mov    %rcx,%r12
  40b41e:	55                   	push   %rbp
  40b41f:	48 89 f5             	mov    %rsi,%rbp
  40b422:	53                   	push   %rbx
  40b423:	48 83 ec 08          	sub    $0x8,%rsp
  40b427:	48 85 d2             	test   %rdx,%rdx
  40b42a:	4c 0f 44 e8          	cmove  %rax,%r13
  40b42e:	48 85 c9             	test   %rcx,%rcx
  40b431:	b8 70 ac 40 00       	mov    $0x40ac70,%eax
  40b436:	4c 0f 44 e0          	cmove  %rax,%r12
  40b43a:	e8 01 72 ff ff       	callq  402640 <malloc@plt>
  40b43f:	48 85 c0             	test   %rax,%rax
  40b442:	48 89 c3             	mov    %rax,%rbx
  40b445:	0f 84 4d 01 00 00    	je     40b598 <__sprintf_chk@plt+0x8d08>
  40b44b:	48 85 ed             	test   %rbp,%rbp
  40b44e:	b8 e0 5e 41 00       	mov    $0x415ee0,%eax
  40b453:	48 8d 7b 28          	lea    0x28(%rbx),%rdi
  40b457:	48 0f 44 e8          	cmove  %rax,%rbp
  40b45b:	48 89 6b 28          	mov    %rbp,0x28(%rbx)
  40b45f:	e8 4c f9 ff ff       	callq  40adb0 <__sprintf_chk@plt+0x8520>
  40b464:	84 c0                	test   %al,%al
  40b466:	0f 84 dc 00 00 00    	je     40b548 <__sprintf_chk@plt+0x8cb8>
  40b46c:	80 7d 10 00          	cmpb   $0x0,0x10(%rbp)
  40b470:	f3 0f 10 4d 08       	movss  0x8(%rbp),%xmm1
  40b475:	75 49                	jne    40b4c0 <__sprintf_chk@plt+0x8c30>
  40b477:	4d 85 ff             	test   %r15,%r15
  40b47a:	0f 88 f8 00 00 00    	js     40b578 <__sprintf_chk@plt+0x8ce8>
  40b480:	f3 49 0f 2a c7       	cvtsi2ss %r15,%xmm0
  40b485:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  40b489:	0f 2e 05 78 aa 00 00 	ucomiss 0xaa78(%rip),%xmm0        # 415f08 <_fini@@Base+0x400c>
  40b490:	0f 83 aa 00 00 00    	jae    40b540 <__sprintf_chk@plt+0x8cb0>
  40b496:	0f 2e 05 6f aa 00 00 	ucomiss 0xaa6f(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40b49d:	0f 82 c5 00 00 00    	jb     40b568 <__sprintf_chk@plt+0x8cd8>
  40b4a3:	f3 0f 5c 05 61 aa 00 	subss  0xaa61(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40b4aa:	00 
  40b4ab:	48 b8 00 00 00 00 00 	movabs $0x8000000000000000,%rax
  40b4b2:	00 00 80 
  40b4b5:	f3 4c 0f 2c f8       	cvttss2si %xmm0,%r15
  40b4ba:	49 31 c7             	xor    %rax,%r15
  40b4bd:	0f 1f 00             	nopl   (%rax)
  40b4c0:	4c 89 ff             	mov    %r15,%rdi
  40b4c3:	e8 f8 f6 ff ff       	callq  40abc0 <__sprintf_chk@plt+0x8330>
  40b4c8:	48 89 c5             	mov    %rax,%rbp
  40b4cb:	48 b8 ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rax
  40b4d2:	ff ff 1f 
  40b4d5:	48 39 c5             	cmp    %rax,%rbp
  40b4d8:	77 66                	ja     40b540 <__sprintf_chk@plt+0x8cb0>
  40b4da:	48 85 ed             	test   %rbp,%rbp
  40b4dd:	48 89 6b 10          	mov    %rbp,0x10(%rbx)
  40b4e1:	74 65                	je     40b548 <__sprintf_chk@plt+0x8cb8>
  40b4e3:	be 10 00 00 00       	mov    $0x10,%esi
  40b4e8:	48 89 ef             	mov    %rbp,%rdi
  40b4eb:	e8 40 70 ff ff       	callq  402530 <calloc@plt>
  40b4f0:	48 85 c0             	test   %rax,%rax
  40b4f3:	48 89 03             	mov    %rax,(%rbx)
  40b4f6:	74 50                	je     40b548 <__sprintf_chk@plt+0x8cb8>
  40b4f8:	48 c1 e5 04          	shl    $0x4,%rbp
  40b4fc:	48 c7 43 18 00 00 00 	movq   $0x0,0x18(%rbx)
  40b503:	00 
  40b504:	48 c7 43 20 00 00 00 	movq   $0x0,0x20(%rbx)
  40b50b:	00 
  40b50c:	48 01 e8             	add    %rbp,%rax
  40b50f:	4c 89 6b 30          	mov    %r13,0x30(%rbx)
  40b513:	4c 89 63 38          	mov    %r12,0x38(%rbx)
  40b517:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40b51b:	4c 89 73 40          	mov    %r14,0x40(%rbx)
  40b51f:	48 89 d8             	mov    %rbx,%rax
  40b522:	48 c7 43 48 00 00 00 	movq   $0x0,0x48(%rbx)
  40b529:	00 
  40b52a:	48 83 c4 08          	add    $0x8,%rsp
  40b52e:	5b                   	pop    %rbx
  40b52f:	5d                   	pop    %rbp
  40b530:	41 5c                	pop    %r12
  40b532:	41 5d                	pop    %r13
  40b534:	41 5e                	pop    %r14
  40b536:	41 5f                	pop    %r15
  40b538:	c3                   	retq   
  40b539:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40b540:	48 c7 43 10 00 00 00 	movq   $0x0,0x10(%rbx)
  40b547:	00 
  40b548:	48 89 df             	mov    %rbx,%rdi
  40b54b:	e8 a0 6c ff ff       	callq  4021f0 <free@plt>
  40b550:	48 83 c4 08          	add    $0x8,%rsp
  40b554:	31 c0                	xor    %eax,%eax
  40b556:	5b                   	pop    %rbx
  40b557:	5d                   	pop    %rbp
  40b558:	41 5c                	pop    %r12
  40b55a:	41 5d                	pop    %r13
  40b55c:	41 5e                	pop    %r14
  40b55e:	41 5f                	pop    %r15
  40b560:	c3                   	retq   
  40b561:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40b568:	f3 4c 0f 2c f8       	cvttss2si %xmm0,%r15
  40b56d:	e9 4e ff ff ff       	jmpq   40b4c0 <__sprintf_chk@plt+0x8c30>
  40b572:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40b578:	4c 89 f8             	mov    %r15,%rax
  40b57b:	41 83 e7 01          	and    $0x1,%r15d
  40b57f:	48 d1 e8             	shr    %rax
  40b582:	4c 09 f8             	or     %r15,%rax
  40b585:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40b58a:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40b58e:	e9 f2 fe ff ff       	jmpq   40b485 <__sprintf_chk@plt+0x8bf5>
  40b593:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40b598:	31 c0                	xor    %eax,%eax
  40b59a:	eb 8e                	jmp    40b52a <__sprintf_chk@plt+0x8c9a>
  40b59c:	0f 1f 40 00          	nopl   0x0(%rax)
  40b5a0:	41 54                	push   %r12
  40b5a2:	55                   	push   %rbp
  40b5a3:	48 89 fd             	mov    %rdi,%rbp
  40b5a6:	53                   	push   %rbx
  40b5a7:	4c 8b 27             	mov    (%rdi),%r12
  40b5aa:	4c 3b 67 08          	cmp    0x8(%rdi),%r12
  40b5ae:	73 73                	jae    40b623 <__sprintf_chk@plt+0x8d93>
  40b5b0:	49 83 3c 24 00       	cmpq   $0x0,(%r12)
  40b5b5:	74 62                	je     40b619 <__sprintf_chk@plt+0x8d89>
  40b5b7:	49 8b 5c 24 08       	mov    0x8(%r12),%rbx
  40b5bc:	48 8b 55 40          	mov    0x40(%rbp),%rdx
  40b5c0:	48 85 db             	test   %rbx,%rbx
  40b5c3:	75 0e                	jne    40b5d3 <__sprintf_chk@plt+0x8d43>
  40b5c5:	eb 36                	jmp    40b5fd <__sprintf_chk@plt+0x8d6d>
  40b5c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40b5ce:	00 00 
  40b5d0:	48 89 c3             	mov    %rax,%rbx
  40b5d3:	48 85 d2             	test   %rdx,%rdx
  40b5d6:	74 09                	je     40b5e1 <__sprintf_chk@plt+0x8d51>
  40b5d8:	48 8b 3b             	mov    (%rbx),%rdi
  40b5db:	ff d2                	callq  *%rdx
  40b5dd:	48 8b 55 40          	mov    0x40(%rbp),%rdx
  40b5e1:	48 8b 43 08          	mov    0x8(%rbx),%rax
  40b5e5:	48 8b 4d 48          	mov    0x48(%rbp),%rcx
  40b5e9:	48 c7 03 00 00 00 00 	movq   $0x0,(%rbx)
  40b5f0:	48 85 c0             	test   %rax,%rax
  40b5f3:	48 89 4b 08          	mov    %rcx,0x8(%rbx)
  40b5f7:	48 89 5d 48          	mov    %rbx,0x48(%rbp)
  40b5fb:	75 d3                	jne    40b5d0 <__sprintf_chk@plt+0x8d40>
  40b5fd:	48 85 d2             	test   %rdx,%rdx
  40b600:	74 06                	je     40b608 <__sprintf_chk@plt+0x8d78>
  40b602:	49 8b 3c 24          	mov    (%r12),%rdi
  40b606:	ff d2                	callq  *%rdx
  40b608:	49 c7 04 24 00 00 00 	movq   $0x0,(%r12)
  40b60f:	00 
  40b610:	49 c7 44 24 08 00 00 	movq   $0x0,0x8(%r12)
  40b617:	00 00 
  40b619:	49 83 c4 10          	add    $0x10,%r12
  40b61d:	4c 39 65 08          	cmp    %r12,0x8(%rbp)
  40b621:	77 8d                	ja     40b5b0 <__sprintf_chk@plt+0x8d20>
  40b623:	48 c7 45 18 00 00 00 	movq   $0x0,0x18(%rbp)
  40b62a:	00 
  40b62b:	48 c7 45 20 00 00 00 	movq   $0x0,0x20(%rbp)
  40b632:	00 
  40b633:	5b                   	pop    %rbx
  40b634:	5d                   	pop    %rbp
  40b635:	41 5c                	pop    %r12
  40b637:	c3                   	retq   
  40b638:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40b63f:	00 
  40b640:	41 54                	push   %r12
  40b642:	55                   	push   %rbp
  40b643:	48 89 fd             	mov    %rdi,%rbp
  40b646:	53                   	push   %rbx
  40b647:	48 83 7f 40 00       	cmpq   $0x0,0x40(%rdi)
  40b64c:	74 07                	je     40b655 <__sprintf_chk@plt+0x8dc5>
  40b64e:	48 83 7f 20 00       	cmpq   $0x0,0x20(%rdi)
  40b653:	75 71                	jne    40b6c6 <__sprintf_chk@plt+0x8e36>
  40b655:	48 8b 45 08          	mov    0x8(%rbp),%rax
  40b659:	4c 8b 65 00          	mov    0x0(%rbp),%r12
  40b65d:	49 39 c4             	cmp    %rax,%r12
  40b660:	73 31                	jae    40b693 <__sprintf_chk@plt+0x8e03>
  40b662:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40b668:	49 8b 7c 24 08       	mov    0x8(%r12),%rdi
  40b66d:	48 85 ff             	test   %rdi,%rdi
  40b670:	75 09                	jne    40b67b <__sprintf_chk@plt+0x8deb>
  40b672:	eb 15                	jmp    40b689 <__sprintf_chk@plt+0x8df9>
  40b674:	0f 1f 40 00          	nopl   0x0(%rax)
  40b678:	48 89 df             	mov    %rbx,%rdi
  40b67b:	48 8b 5f 08          	mov    0x8(%rdi),%rbx
  40b67f:	e8 6c 6b ff ff       	callq  4021f0 <free@plt>
  40b684:	48 85 db             	test   %rbx,%rbx
  40b687:	75 ef                	jne    40b678 <__sprintf_chk@plt+0x8de8>
  40b689:	49 83 c4 10          	add    $0x10,%r12
  40b68d:	4c 39 65 08          	cmp    %r12,0x8(%rbp)
  40b691:	77 d5                	ja     40b668 <__sprintf_chk@plt+0x8dd8>
  40b693:	48 8b 7d 48          	mov    0x48(%rbp),%rdi
  40b697:	48 85 ff             	test   %rdi,%rdi
  40b69a:	75 07                	jne    40b6a3 <__sprintf_chk@plt+0x8e13>
  40b69c:	eb 13                	jmp    40b6b1 <__sprintf_chk@plt+0x8e21>
  40b69e:	66 90                	xchg   %ax,%ax
  40b6a0:	48 89 df             	mov    %rbx,%rdi
  40b6a3:	48 8b 5f 08          	mov    0x8(%rdi),%rbx
  40b6a7:	e8 44 6b ff ff       	callq  4021f0 <free@plt>
  40b6ac:	48 85 db             	test   %rbx,%rbx
  40b6af:	75 ef                	jne    40b6a0 <__sprintf_chk@plt+0x8e10>
  40b6b1:	48 8b 7d 00          	mov    0x0(%rbp),%rdi
  40b6b5:	e8 36 6b ff ff       	callq  4021f0 <free@plt>
  40b6ba:	5b                   	pop    %rbx
  40b6bb:	48 89 ef             	mov    %rbp,%rdi
  40b6be:	5d                   	pop    %rbp
  40b6bf:	41 5c                	pop    %r12
  40b6c1:	e9 2a 6b ff ff       	jmpq   4021f0 <free@plt>
  40b6c6:	4c 8b 27             	mov    (%rdi),%r12
  40b6c9:	4c 3b 67 08          	cmp    0x8(%rdi),%r12
  40b6cd:	73 c4                	jae    40b693 <__sprintf_chk@plt+0x8e03>
  40b6cf:	90                   	nop
  40b6d0:	49 8b 3c 24          	mov    (%r12),%rdi
  40b6d4:	4c 89 e3             	mov    %r12,%rbx
  40b6d7:	48 85 ff             	test   %rdi,%rdi
  40b6da:	75 07                	jne    40b6e3 <__sprintf_chk@plt+0x8e53>
  40b6dc:	eb 11                	jmp    40b6ef <__sprintf_chk@plt+0x8e5f>
  40b6de:	66 90                	xchg   %ax,%ax
  40b6e0:	48 8b 3b             	mov    (%rbx),%rdi
  40b6e3:	ff 55 40             	callq  *0x40(%rbp)
  40b6e6:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40b6ea:	48 85 db             	test   %rbx,%rbx
  40b6ed:	75 f1                	jne    40b6e0 <__sprintf_chk@plt+0x8e50>
  40b6ef:	48 8b 45 08          	mov    0x8(%rbp),%rax
  40b6f3:	49 83 c4 10          	add    $0x10,%r12
  40b6f7:	4c 39 e0             	cmp    %r12,%rax
  40b6fa:	77 d4                	ja     40b6d0 <__sprintf_chk@plt+0x8e40>
  40b6fc:	e9 58 ff ff ff       	jmpq   40b659 <__sprintf_chk@plt+0x8dc9>
  40b701:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40b708:	0f 1f 84 00 00 00 00 
  40b70f:	00 
  40b710:	41 54                	push   %r12
  40b712:	55                   	push   %rbp
  40b713:	53                   	push   %rbx
  40b714:	48 89 fb             	mov    %rdi,%rbx
  40b717:	48 83 ec 50          	sub    $0x50,%rsp
  40b71b:	48 8b 6f 28          	mov    0x28(%rdi),%rbp
  40b71f:	80 7d 10 00          	cmpb   $0x0,0x10(%rbp)
  40b723:	f3 0f 10 4d 08       	movss  0x8(%rbp),%xmm1
  40b728:	75 46                	jne    40b770 <__sprintf_chk@plt+0x8ee0>
  40b72a:	48 85 f6             	test   %rsi,%rsi
  40b72d:	0f 88 9d 01 00 00    	js     40b8d0 <__sprintf_chk@plt+0x9040>
  40b733:	f3 48 0f 2a c6       	cvtsi2ss %rsi,%xmm0
  40b738:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  40b73c:	0f 2e 05 c5 a7 00 00 	ucomiss 0xa7c5(%rip),%xmm0        # 415f08 <_fini@@Base+0x400c>
  40b743:	0f 83 27 01 00 00    	jae    40b870 <__sprintf_chk@plt+0x8fe0>
  40b749:	0f 2e 05 bc a7 00 00 	ucomiss 0xa7bc(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40b750:	0f 82 2a 01 00 00    	jb     40b880 <__sprintf_chk@plt+0x8ff0>
  40b756:	f3 0f 5c 05 ae a7 00 	subss  0xa7ae(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40b75d:	00 
  40b75e:	48 b8 00 00 00 00 00 	movabs $0x8000000000000000,%rax
  40b765:	00 00 80 
  40b768:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40b76d:	48 31 c6             	xor    %rax,%rsi
  40b770:	48 89 f7             	mov    %rsi,%rdi
  40b773:	e8 48 f4 ff ff       	callq  40abc0 <__sprintf_chk@plt+0x8330>
  40b778:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
  40b77c:	49 89 c4             	mov    %rax,%r12
  40b77f:	48 b8 fe ff ff ff ff 	movabs $0x1ffffffffffffffe,%rax
  40b786:	ff ff 1f 
  40b789:	48 39 c2             	cmp    %rax,%rdx
  40b78c:	0f 87 de 00 00 00    	ja     40b870 <__sprintf_chk@plt+0x8fe0>
  40b792:	4c 39 63 10          	cmp    %r12,0x10(%rbx)
  40b796:	0f 84 c4 00 00 00    	je     40b860 <__sprintf_chk@plt+0x8fd0>
  40b79c:	be 10 00 00 00       	mov    $0x10,%esi
  40b7a1:	4c 89 e7             	mov    %r12,%rdi
  40b7a4:	e8 87 6d ff ff       	callq  402530 <calloc@plt>
  40b7a9:	48 85 c0             	test   %rax,%rax
  40b7ac:	48 89 04 24          	mov    %rax,(%rsp)
  40b7b0:	0f 84 ba 00 00 00    	je     40b870 <__sprintf_chk@plt+0x8fe0>
  40b7b6:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  40b7bb:	49 c1 e4 04          	shl    $0x4,%r12
  40b7bf:	31 d2                	xor    %edx,%edx
  40b7c1:	4c 01 e0             	add    %r12,%rax
  40b7c4:	48 89 de             	mov    %rbx,%rsi
  40b7c7:	48 89 e7             	mov    %rsp,%rdi
  40b7ca:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40b7cf:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40b7d3:	48 89 6c 24 28       	mov    %rbp,0x28(%rsp)
  40b7d8:	48 c7 44 24 18 00 00 	movq   $0x0,0x18(%rsp)
  40b7df:	00 00 
  40b7e1:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
  40b7e8:	00 00 
  40b7ea:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  40b7ef:	48 8b 43 38          	mov    0x38(%rbx),%rax
  40b7f3:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  40b7f8:	48 8b 43 40          	mov    0x40(%rbx),%rax
  40b7fc:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  40b801:	48 8b 43 48          	mov    0x48(%rbx),%rax
  40b805:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40b80a:	e8 31 f6 ff ff       	callq  40ae40 <__sprintf_chk@plt+0x85b0>
  40b80f:	84 c0                	test   %al,%al
  40b811:	89 c5                	mov    %eax,%ebp
  40b813:	75 7b                	jne    40b890 <__sprintf_chk@plt+0x9000>
  40b815:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40b81a:	ba 01 00 00 00       	mov    $0x1,%edx
  40b81f:	48 89 e6             	mov    %rsp,%rsi
  40b822:	48 89 df             	mov    %rbx,%rdi
  40b825:	48 89 43 48          	mov    %rax,0x48(%rbx)
  40b829:	e8 12 f6 ff ff       	callq  40ae40 <__sprintf_chk@plt+0x85b0>
  40b82e:	84 c0                	test   %al,%al
  40b830:	0f 84 b4 00 00 00    	je     40b8ea <__sprintf_chk@plt+0x905a>
  40b836:	31 d2                	xor    %edx,%edx
  40b838:	48 89 e6             	mov    %rsp,%rsi
  40b83b:	48 89 df             	mov    %rbx,%rdi
  40b83e:	e8 fd f5 ff ff       	callq  40ae40 <__sprintf_chk@plt+0x85b0>
  40b843:	84 c0                	test   %al,%al
  40b845:	0f 84 9f 00 00 00    	je     40b8ea <__sprintf_chk@plt+0x905a>
  40b84b:	48 8b 3c 24          	mov    (%rsp),%rdi
  40b84f:	e8 9c 69 ff ff       	callq  4021f0 <free@plt>
  40b854:	48 83 c4 50          	add    $0x50,%rsp
  40b858:	89 e8                	mov    %ebp,%eax
  40b85a:	5b                   	pop    %rbx
  40b85b:	5d                   	pop    %rbp
  40b85c:	41 5c                	pop    %r12
  40b85e:	c3                   	retq   
  40b85f:	90                   	nop
  40b860:	48 83 c4 50          	add    $0x50,%rsp
  40b864:	bd 01 00 00 00       	mov    $0x1,%ebp
  40b869:	5b                   	pop    %rbx
  40b86a:	89 e8                	mov    %ebp,%eax
  40b86c:	5d                   	pop    %rbp
  40b86d:	41 5c                	pop    %r12
  40b86f:	c3                   	retq   
  40b870:	48 83 c4 50          	add    $0x50,%rsp
  40b874:	31 ed                	xor    %ebp,%ebp
  40b876:	5b                   	pop    %rbx
  40b877:	89 e8                	mov    %ebp,%eax
  40b879:	5d                   	pop    %rbp
  40b87a:	41 5c                	pop    %r12
  40b87c:	c3                   	retq   
  40b87d:	0f 1f 00             	nopl   (%rax)
  40b880:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40b885:	e9 e6 fe ff ff       	jmpq   40b770 <__sprintf_chk@plt+0x8ee0>
  40b88a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40b890:	48 8b 3b             	mov    (%rbx),%rdi
  40b893:	e8 58 69 ff ff       	callq  4021f0 <free@plt>
  40b898:	48 8b 04 24          	mov    (%rsp),%rax
  40b89c:	48 89 03             	mov    %rax,(%rbx)
  40b89f:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40b8a4:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40b8a8:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40b8ad:	48 89 43 10          	mov    %rax,0x10(%rbx)
  40b8b1:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40b8b6:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40b8ba:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40b8bf:	48 89 43 48          	mov    %rax,0x48(%rbx)
  40b8c3:	48 83 c4 50          	add    $0x50,%rsp
  40b8c7:	89 e8                	mov    %ebp,%eax
  40b8c9:	5b                   	pop    %rbx
  40b8ca:	5d                   	pop    %rbp
  40b8cb:	41 5c                	pop    %r12
  40b8cd:	c3                   	retq   
  40b8ce:	66 90                	xchg   %ax,%ax
  40b8d0:	48 89 f0             	mov    %rsi,%rax
  40b8d3:	83 e6 01             	and    $0x1,%esi
  40b8d6:	48 d1 e8             	shr    %rax
  40b8d9:	48 09 f0             	or     %rsi,%rax
  40b8dc:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40b8e1:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40b8e5:	e9 4e fe ff ff       	jmpq   40b738 <__sprintf_chk@plt+0x8ea8>
  40b8ea:	e8 31 69 ff ff       	callq  402220 <abort@plt>
  40b8ef:	90                   	nop
  40b8f0:	41 54                	push   %r12
  40b8f2:	55                   	push   %rbp
  40b8f3:	48 89 f5             	mov    %rsi,%rbp
  40b8f6:	53                   	push   %rbx
  40b8f7:	48 83 ec 10          	sub    $0x10,%rsp
  40b8fb:	48 85 f6             	test   %rsi,%rsi
  40b8fe:	0f 84 63 01 00 00    	je     40ba67 <__sprintf_chk@plt+0x91d7>
  40b904:	49 89 d4             	mov    %rdx,%r12
  40b907:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40b90c:	31 c9                	xor    %ecx,%ecx
  40b90e:	48 89 fb             	mov    %rdi,%rbx
  40b911:	e8 9a f3 ff ff       	callq  40acb0 <__sprintf_chk@plt+0x8420>
  40b916:	48 85 c0             	test   %rax,%rax
  40b919:	74 1d                	je     40b938 <__sprintf_chk@plt+0x90a8>
  40b91b:	4d 85 e4             	test   %r12,%r12
  40b91e:	0f 84 94 00 00 00    	je     40b9b8 <__sprintf_chk@plt+0x9128>
  40b924:	49 89 04 24          	mov    %rax,(%r12)
  40b928:	31 c0                	xor    %eax,%eax
  40b92a:	48 83 c4 10          	add    $0x10,%rsp
  40b92e:	5b                   	pop    %rbx
  40b92f:	5d                   	pop    %rbp
  40b930:	41 5c                	pop    %r12
  40b932:	c3                   	retq   
  40b933:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40b938:	48 8b 43 18          	mov    0x18(%rbx),%rax
  40b93c:	48 85 c0             	test   %rax,%rax
  40b93f:	0f 88 2b 01 00 00    	js     40ba70 <__sprintf_chk@plt+0x91e0>
  40b945:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40b94a:	48 8b 43 10          	mov    0x10(%rbx),%rax
  40b94e:	48 8b 53 28          	mov    0x28(%rbx),%rdx
  40b952:	48 85 c0             	test   %rax,%rax
  40b955:	0f 88 35 01 00 00    	js     40ba90 <__sprintf_chk@plt+0x9200>
  40b95b:	f3 48 0f 2a c8       	cvtsi2ss %rax,%xmm1
  40b960:	f3 0f 59 4a 08       	mulss  0x8(%rdx),%xmm1
  40b965:	0f 2e c1             	ucomiss %xmm1,%xmm0
  40b968:	77 5e                	ja     40b9c8 <__sprintf_chk@plt+0x9138>
  40b96a:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
  40b96f:	49 83 3c 24 00       	cmpq   $0x0,(%r12)
  40b974:	0f 84 36 01 00 00    	je     40bab0 <__sprintf_chk@plt+0x9220>
  40b97a:	48 8b 43 48          	mov    0x48(%rbx),%rax
  40b97e:	48 85 c0             	test   %rax,%rax
  40b981:	0f 84 94 01 00 00    	je     40bb1b <__sprintf_chk@plt+0x928b>
  40b987:	48 8b 50 08          	mov    0x8(%rax),%rdx
  40b98b:	48 89 53 48          	mov    %rdx,0x48(%rbx)
  40b98f:	49 8b 54 24 08       	mov    0x8(%r12),%rdx
  40b994:	48 89 28             	mov    %rbp,(%rax)
  40b997:	48 89 50 08          	mov    %rdx,0x8(%rax)
  40b99b:	49 89 44 24 08       	mov    %rax,0x8(%r12)
  40b9a0:	b8 01 00 00 00       	mov    $0x1,%eax
  40b9a5:	48 83 43 20 01       	addq   $0x1,0x20(%rbx)
  40b9aa:	48 83 c4 10          	add    $0x10,%rsp
  40b9ae:	5b                   	pop    %rbx
  40b9af:	5d                   	pop    %rbp
  40b9b0:	41 5c                	pop    %r12
  40b9b2:	c3                   	retq   
  40b9b3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40b9b8:	48 83 c4 10          	add    $0x10,%rsp
  40b9bc:	31 c0                	xor    %eax,%eax
  40b9be:	5b                   	pop    %rbx
  40b9bf:	5d                   	pop    %rbp
  40b9c0:	41 5c                	pop    %r12
  40b9c2:	c3                   	retq   
  40b9c3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40b9c8:	48 8d 7b 28          	lea    0x28(%rbx),%rdi
  40b9cc:	e8 df f3 ff ff       	callq  40adb0 <__sprintf_chk@plt+0x8520>
  40b9d1:	48 8b 43 10          	mov    0x10(%rbx),%rax
  40b9d5:	48 8b 53 28          	mov    0x28(%rbx),%rdx
  40b9d9:	48 85 c0             	test   %rax,%rax
  40b9dc:	f3 0f 10 52 08       	movss  0x8(%rdx),%xmm2
  40b9e1:	0f 88 00 01 00 00    	js     40bae7 <__sprintf_chk@plt+0x9257>
  40b9e7:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40b9ec:	48 8b 43 18          	mov    0x18(%rbx),%rax
  40b9f0:	48 85 c0             	test   %rax,%rax
  40b9f3:	0f 88 08 01 00 00    	js     40bb01 <__sprintf_chk@plt+0x9271>
  40b9f9:	f3 48 0f 2a c8       	cvtsi2ss %rax,%xmm1
  40b9fe:	0f 28 da             	movaps %xmm2,%xmm3
  40ba01:	f3 0f 59 d8          	mulss  %xmm0,%xmm3
  40ba05:	0f 2e cb             	ucomiss %xmm3,%xmm1
  40ba08:	0f 86 5c ff ff ff    	jbe    40b96a <__sprintf_chk@plt+0x90da>
  40ba0e:	80 7a 10 00          	cmpb   $0x0,0x10(%rdx)
  40ba12:	f3 0f 59 42 0c       	mulss  0xc(%rdx),%xmm0
  40ba17:	75 04                	jne    40ba1d <__sprintf_chk@plt+0x918d>
  40ba19:	f3 0f 59 c2          	mulss  %xmm2,%xmm0
  40ba1d:	0f 2e 05 e4 a4 00 00 	ucomiss 0xa4e4(%rip),%xmm0        # 415f08 <_fini@@Base+0x400c>
  40ba24:	0f 83 04 01 00 00    	jae    40bb2e <__sprintf_chk@plt+0x929e>
  40ba2a:	0f 2e 05 db a4 00 00 	ucomiss 0xa4db(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40ba31:	0f 83 91 00 00 00    	jae    40bac8 <__sprintf_chk@plt+0x9238>
  40ba37:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40ba3c:	48 89 df             	mov    %rbx,%rdi
  40ba3f:	e8 cc fc ff ff       	callq  40b710 <__sprintf_chk@plt+0x8e80>
  40ba44:	84 c0                	test   %al,%al
  40ba46:	0f 84 e2 00 00 00    	je     40bb2e <__sprintf_chk@plt+0x929e>
  40ba4c:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40ba51:	31 c9                	xor    %ecx,%ecx
  40ba53:	48 89 ee             	mov    %rbp,%rsi
  40ba56:	48 89 df             	mov    %rbx,%rdi
  40ba59:	e8 52 f2 ff ff       	callq  40acb0 <__sprintf_chk@plt+0x8420>
  40ba5e:	48 85 c0             	test   %rax,%rax
  40ba61:	0f 84 03 ff ff ff    	je     40b96a <__sprintf_chk@plt+0x90da>
  40ba67:	e8 b4 67 ff ff       	callq  402220 <abort@plt>
  40ba6c:	0f 1f 40 00          	nopl   0x0(%rax)
  40ba70:	48 89 c2             	mov    %rax,%rdx
  40ba73:	83 e0 01             	and    $0x1,%eax
  40ba76:	48 d1 ea             	shr    %rdx
  40ba79:	48 09 c2             	or     %rax,%rdx
  40ba7c:	f3 48 0f 2a c2       	cvtsi2ss %rdx,%xmm0
  40ba81:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40ba85:	e9 c0 fe ff ff       	jmpq   40b94a <__sprintf_chk@plt+0x90ba>
  40ba8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ba90:	48 89 c1             	mov    %rax,%rcx
  40ba93:	83 e0 01             	and    $0x1,%eax
  40ba96:	48 d1 e9             	shr    %rcx
  40ba99:	48 09 c1             	or     %rax,%rcx
  40ba9c:	f3 48 0f 2a c9       	cvtsi2ss %rcx,%xmm1
  40baa1:	f3 0f 58 c9          	addss  %xmm1,%xmm1
  40baa5:	e9 b6 fe ff ff       	jmpq   40b960 <__sprintf_chk@plt+0x90d0>
  40baaa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40bab0:	49 89 2c 24          	mov    %rbp,(%r12)
  40bab4:	b8 01 00 00 00       	mov    $0x1,%eax
  40bab9:	48 83 43 20 01       	addq   $0x1,0x20(%rbx)
  40babe:	48 83 43 18 01       	addq   $0x1,0x18(%rbx)
  40bac3:	e9 62 fe ff ff       	jmpq   40b92a <__sprintf_chk@plt+0x909a>
  40bac8:	f3 0f 5c 05 3c a4 00 	subss  0xa43c(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40bacf:	00 
  40bad0:	48 b8 00 00 00 00 00 	movabs $0x8000000000000000,%rax
  40bad7:	00 00 80 
  40bada:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40badf:	48 31 c6             	xor    %rax,%rsi
  40bae2:	e9 55 ff ff ff       	jmpq   40ba3c <__sprintf_chk@plt+0x91ac>
  40bae7:	48 89 c1             	mov    %rax,%rcx
  40baea:	83 e0 01             	and    $0x1,%eax
  40baed:	48 d1 e9             	shr    %rcx
  40baf0:	48 09 c1             	or     %rax,%rcx
  40baf3:	f3 48 0f 2a c1       	cvtsi2ss %rcx,%xmm0
  40baf8:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40bafc:	e9 eb fe ff ff       	jmpq   40b9ec <__sprintf_chk@plt+0x915c>
  40bb01:	48 89 c1             	mov    %rax,%rcx
  40bb04:	83 e0 01             	and    $0x1,%eax
  40bb07:	48 d1 e9             	shr    %rcx
  40bb0a:	48 09 c1             	or     %rax,%rcx
  40bb0d:	f3 48 0f 2a c9       	cvtsi2ss %rcx,%xmm1
  40bb12:	f3 0f 58 c9          	addss  %xmm1,%xmm1
  40bb16:	e9 e3 fe ff ff       	jmpq   40b9fe <__sprintf_chk@plt+0x916e>
  40bb1b:	bf 10 00 00 00       	mov    $0x10,%edi
  40bb20:	e8 1b 6b ff ff       	callq  402640 <malloc@plt>
  40bb25:	48 85 c0             	test   %rax,%rax
  40bb28:	0f 85 61 fe ff ff    	jne    40b98f <__sprintf_chk@plt+0x90ff>
  40bb2e:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40bb33:	e9 f2 fd ff ff       	jmpq   40b92a <__sprintf_chk@plt+0x909a>
  40bb38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40bb3f:	00 
  40bb40:	e9 ab fd ff ff       	jmpq   40b8f0 <__sprintf_chk@plt+0x9060>
  40bb45:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40bb4c:	00 00 00 00 
  40bb50:	53                   	push   %rbx
  40bb51:	48 89 f3             	mov    %rsi,%rbx
  40bb54:	48 83 ec 10          	sub    $0x10,%rsp
  40bb58:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40bb5d:	e8 8e fd ff ff       	callq  40b8f0 <__sprintf_chk@plt+0x9060>
  40bb62:	83 f8 ff             	cmp    $0xffffffff,%eax
  40bb65:	74 19                	je     40bb80 <__sprintf_chk@plt+0x92f0>
  40bb67:	85 c0                	test   %eax,%eax
  40bb69:	48 89 d8             	mov    %rbx,%rax
  40bb6c:	48 0f 44 44 24 08    	cmove  0x8(%rsp),%rax
  40bb72:	48 83 c4 10          	add    $0x10,%rsp
  40bb76:	5b                   	pop    %rbx
  40bb77:	c3                   	retq   
  40bb78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40bb7f:	00 
  40bb80:	31 c0                	xor    %eax,%eax
  40bb82:	eb ee                	jmp    40bb72 <__sprintf_chk@plt+0x92e2>
  40bb84:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40bb8b:	00 00 00 00 00 
  40bb90:	41 54                	push   %r12
  40bb92:	b9 01 00 00 00       	mov    $0x1,%ecx
  40bb97:	55                   	push   %rbp
  40bb98:	53                   	push   %rbx
  40bb99:	48 89 fb             	mov    %rdi,%rbx
  40bb9c:	48 83 ec 10          	sub    $0x10,%rsp
  40bba0:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx
  40bba5:	e8 06 f1 ff ff       	callq  40acb0 <__sprintf_chk@plt+0x8420>
  40bbaa:	48 85 c0             	test   %rax,%rax
  40bbad:	48 89 c5             	mov    %rax,%rbp
  40bbb0:	0f 84 52 01 00 00    	je     40bd08 <__sprintf_chk@plt+0x9478>
  40bbb6:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  40bbbb:	48 83 6b 20 01       	subq   $0x1,0x20(%rbx)
  40bbc0:	48 83 3a 00          	cmpq   $0x0,(%rdx)
  40bbc4:	74 0a                	je     40bbd0 <__sprintf_chk@plt+0x9340>
  40bbc6:	48 83 c4 10          	add    $0x10,%rsp
  40bbca:	5b                   	pop    %rbx
  40bbcb:	5d                   	pop    %rbp
  40bbcc:	41 5c                	pop    %r12
  40bbce:	c3                   	retq   
  40bbcf:	90                   	nop
  40bbd0:	48 8b 43 18          	mov    0x18(%rbx),%rax
  40bbd4:	48 83 e8 01          	sub    $0x1,%rax
  40bbd8:	48 85 c0             	test   %rax,%rax
  40bbdb:	48 89 43 18          	mov    %rax,0x18(%rbx)
  40bbdf:	0f 88 03 01 00 00    	js     40bce8 <__sprintf_chk@plt+0x9458>
  40bbe5:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40bbea:	48 8b 43 10          	mov    0x10(%rbx),%rax
  40bbee:	48 8b 53 28          	mov    0x28(%rbx),%rdx
  40bbf2:	48 85 c0             	test   %rax,%rax
  40bbf5:	0f 88 cd 00 00 00    	js     40bcc8 <__sprintf_chk@plt+0x9438>
  40bbfb:	f3 48 0f 2a c8       	cvtsi2ss %rax,%xmm1
  40bc00:	f3 0f 59 0a          	mulss  (%rdx),%xmm1
  40bc04:	0f 2e c8             	ucomiss %xmm0,%xmm1
  40bc07:	77 0f                	ja     40bc18 <__sprintf_chk@plt+0x9388>
  40bc09:	48 83 c4 10          	add    $0x10,%rsp
  40bc0d:	48 89 e8             	mov    %rbp,%rax
  40bc10:	5b                   	pop    %rbx
  40bc11:	5d                   	pop    %rbp
  40bc12:	41 5c                	pop    %r12
  40bc14:	c3                   	retq   
  40bc15:	0f 1f 00             	nopl   (%rax)
  40bc18:	48 8d 7b 28          	lea    0x28(%rbx),%rdi
  40bc1c:	e8 8f f1 ff ff       	callq  40adb0 <__sprintf_chk@plt+0x8520>
  40bc21:	48 8b 53 10          	mov    0x10(%rbx),%rdx
  40bc25:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40bc29:	48 85 d2             	test   %rdx,%rdx
  40bc2c:	0f 88 fd 00 00 00    	js     40bd2f <__sprintf_chk@plt+0x949f>
  40bc32:	f3 48 0f 2a c2       	cvtsi2ss %rdx,%xmm0
  40bc37:	48 8b 53 18          	mov    0x18(%rbx),%rdx
  40bc3b:	48 85 d2             	test   %rdx,%rdx
  40bc3e:	0f 88 05 01 00 00    	js     40bd49 <__sprintf_chk@plt+0x94b9>
  40bc44:	f3 48 0f 2a ca       	cvtsi2ss %rdx,%xmm1
  40bc49:	f3 0f 10 10          	movss  (%rax),%xmm2
  40bc4d:	f3 0f 59 d0          	mulss  %xmm0,%xmm2
  40bc51:	0f 2e d1             	ucomiss %xmm1,%xmm2
  40bc54:	76 b3                	jbe    40bc09 <__sprintf_chk@plt+0x9379>
  40bc56:	80 78 10 00          	cmpb   $0x0,0x10(%rax)
  40bc5a:	f3 0f 59 40 04       	mulss  0x4(%rax),%xmm0
  40bc5f:	75 05                	jne    40bc66 <__sprintf_chk@plt+0x93d6>
  40bc61:	f3 0f 59 40 08       	mulss  0x8(%rax),%xmm0
  40bc66:	0f 2e 05 9f a2 00 00 	ucomiss 0xa29f(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40bc6d:	0f 83 9d 00 00 00    	jae    40bd10 <__sprintf_chk@plt+0x9480>
  40bc73:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40bc78:	48 89 df             	mov    %rbx,%rdi
  40bc7b:	e8 90 fa ff ff       	callq  40b710 <__sprintf_chk@plt+0x8e80>
  40bc80:	89 c2                	mov    %eax,%edx
  40bc82:	48 89 e8             	mov    %rbp,%rax
  40bc85:	84 d2                	test   %dl,%dl
  40bc87:	0f 85 39 ff ff ff    	jne    40bbc6 <__sprintf_chk@plt+0x9336>
  40bc8d:	48 8b 7b 48          	mov    0x48(%rbx),%rdi
  40bc91:	48 85 ff             	test   %rdi,%rdi
  40bc94:	75 0d                	jne    40bca3 <__sprintf_chk@plt+0x9413>
  40bc96:	eb 19                	jmp    40bcb1 <__sprintf_chk@plt+0x9421>
  40bc98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40bc9f:	00 
  40bca0:	4c 89 e7             	mov    %r12,%rdi
  40bca3:	4c 8b 67 08          	mov    0x8(%rdi),%r12
  40bca7:	e8 44 65 ff ff       	callq  4021f0 <free@plt>
  40bcac:	4d 85 e4             	test   %r12,%r12
  40bcaf:	75 ef                	jne    40bca0 <__sprintf_chk@plt+0x9410>
  40bcb1:	48 c7 43 48 00 00 00 	movq   $0x0,0x48(%rbx)
  40bcb8:	00 
  40bcb9:	48 89 e8             	mov    %rbp,%rax
  40bcbc:	e9 05 ff ff ff       	jmpq   40bbc6 <__sprintf_chk@plt+0x9336>
  40bcc1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40bcc8:	48 89 c1             	mov    %rax,%rcx
  40bccb:	83 e0 01             	and    $0x1,%eax
  40bcce:	48 d1 e9             	shr    %rcx
  40bcd1:	48 09 c1             	or     %rax,%rcx
  40bcd4:	f3 48 0f 2a c9       	cvtsi2ss %rcx,%xmm1
  40bcd9:	f3 0f 58 c9          	addss  %xmm1,%xmm1
  40bcdd:	e9 1e ff ff ff       	jmpq   40bc00 <__sprintf_chk@plt+0x9370>
  40bce2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40bce8:	48 89 c2             	mov    %rax,%rdx
  40bceb:	83 e0 01             	and    $0x1,%eax
  40bcee:	48 d1 ea             	shr    %rdx
  40bcf1:	48 09 c2             	or     %rax,%rdx
  40bcf4:	f3 48 0f 2a c2       	cvtsi2ss %rdx,%xmm0
  40bcf9:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40bcfd:	e9 e8 fe ff ff       	jmpq   40bbea <__sprintf_chk@plt+0x935a>
  40bd02:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40bd08:	31 c0                	xor    %eax,%eax
  40bd0a:	e9 b7 fe ff ff       	jmpq   40bbc6 <__sprintf_chk@plt+0x9336>
  40bd0f:	90                   	nop
  40bd10:	f3 0f 5c 05 f4 a1 00 	subss  0xa1f4(%rip),%xmm0        # 415f0c <_fini@@Base+0x4010>
  40bd17:	00 
  40bd18:	48 b8 00 00 00 00 00 	movabs $0x8000000000000000,%rax
  40bd1f:	00 00 80 
  40bd22:	f3 48 0f 2c f0       	cvttss2si %xmm0,%rsi
  40bd27:	48 31 c6             	xor    %rax,%rsi
  40bd2a:	e9 49 ff ff ff       	jmpq   40bc78 <__sprintf_chk@plt+0x93e8>
  40bd2f:	48 89 d1             	mov    %rdx,%rcx
  40bd32:	83 e2 01             	and    $0x1,%edx
  40bd35:	48 d1 e9             	shr    %rcx
  40bd38:	48 09 d1             	or     %rdx,%rcx
  40bd3b:	f3 48 0f 2a c1       	cvtsi2ss %rcx,%xmm0
  40bd40:	f3 0f 58 c0          	addss  %xmm0,%xmm0
  40bd44:	e9 ee fe ff ff       	jmpq   40bc37 <__sprintf_chk@plt+0x93a7>
  40bd49:	48 89 d1             	mov    %rdx,%rcx
  40bd4c:	83 e2 01             	and    $0x1,%edx
  40bd4f:	48 d1 e9             	shr    %rcx
  40bd52:	48 09 d1             	or     %rdx,%rcx
  40bd55:	f3 48 0f 2a c9       	cvtsi2ss %rcx,%xmm1
  40bd5a:	f3 0f 58 c9          	addss  %xmm1,%xmm1
  40bd5e:	e9 e6 fe ff ff       	jmpq   40bc49 <__sprintf_chk@plt+0x93b9>
  40bd63:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40bd6a:	00 00 00 
  40bd6d:	0f 1f 00             	nopl   (%rax)
  40bd70:	41 57                	push   %r15
  40bd72:	89 d0                	mov    %edx,%eax
  40bd74:	83 e0 03             	and    $0x3,%eax
  40bd77:	41 56                	push   %r14
  40bd79:	49 89 f6             	mov    %rsi,%r14
  40bd7c:	41 55                	push   %r13
  40bd7e:	41 54                	push   %r12
  40bd80:	49 89 fc             	mov    %rdi,%r12
  40bd83:	55                   	push   %rbp
  40bd84:	53                   	push   %rbx
  40bd85:	48 89 cb             	mov    %rcx,%rbx
  40bd88:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  40bd8f:	89 44 24 38          	mov    %eax,0x38(%rsp)
  40bd93:	89 d0                	mov    %edx,%eax
  40bd95:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
  40bd9a:	83 e0 20             	and    $0x20,%eax
  40bd9d:	89 54 24 20          	mov    %edx,0x20(%rsp)
  40bda1:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
  40bda6:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  40bdad:	00 00 
  40bdaf:	48 89 8c 24 a8 00 00 	mov    %rcx,0xa8(%rsp)
  40bdb6:	00 
  40bdb7:	31 c9                	xor    %ecx,%ecx
  40bdb9:	83 f8 01             	cmp    $0x1,%eax
  40bdbc:	89 44 24 58          	mov    %eax,0x58(%rsp)
  40bdc0:	19 c0                	sbb    %eax,%eax
  40bdc2:	89 44 24 24          	mov    %eax,0x24(%rsp)
  40bdc6:	83 64 24 24 e8       	andl   $0xffffffe8,0x24(%rsp)
  40bdcb:	81 44 24 24 00 04 00 	addl   $0x400,0x24(%rsp)
  40bdd2:	00 
  40bdd3:	e8 e8 64 ff ff       	callq  4022c0 <localeconv@plt>
  40bdd8:	4c 8b 38             	mov    (%rax),%r15
  40bddb:	49 89 c5             	mov    %rax,%r13
  40bdde:	4c 89 ff             	mov    %r15,%rdi
  40bde1:	e8 9a 65 ff ff       	callq  402380 <strlen@plt>
  40bde6:	49 8b 6d 10          	mov    0x10(%r13),%rbp
  40bdea:	49 89 c3             	mov    %rax,%r11
  40bded:	4d 8b 6d 08          	mov    0x8(%r13),%r13
  40bdf1:	48 8d 40 ff          	lea    -0x1(%rax),%rax
  40bdf5:	ba 01 00 00 00       	mov    $0x1,%edx
  40bdfa:	48 83 f8 10          	cmp    $0x10,%rax
  40bdfe:	4c 89 ef             	mov    %r13,%rdi
  40be01:	b8 90 39 41 00       	mov    $0x413990,%eax
  40be06:	4c 0f 43 da          	cmovae %rdx,%r11
  40be0a:	4c 0f 43 f8          	cmovae %rax,%r15
  40be0e:	4c 89 5c 24 50       	mov    %r11,0x50(%rsp)
  40be13:	e8 68 65 ff ff       	callq  402380 <strlen@plt>
  40be18:	48 83 f8 11          	cmp    $0x11,%rax
  40be1c:	b8 19 69 41 00       	mov    $0x416919,%eax
  40be21:	4c 8b 5c 24 50       	mov    0x50(%rsp),%r11
  40be26:	4c 0f 43 e8          	cmovae %rax,%r13
  40be2a:	4c 89 f0             	mov    %r14,%rax
  40be2d:	48 05 88 02 00 00    	add    $0x288,%rax
  40be33:	48 39 5c 24 28       	cmp    %rbx,0x28(%rsp)
  40be38:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40be3d:	0f 87 9d 02 00 00    	ja     40c0e0 <__sprintf_chk@plt+0x9850>
  40be43:	31 d2                	xor    %edx,%edx
  40be45:	48 89 d8             	mov    %rbx,%rax
  40be48:	48 f7 74 24 28       	divq   0x28(%rsp)
  40be4d:	48 85 d2             	test   %rdx,%rdx
  40be50:	48 89 c1             	mov    %rax,%rcx
  40be53:	0f 84 97 04 00 00    	je     40c2f0 <__sprintf_chk@plt+0x9a60>
  40be59:	4c 89 64 24 68       	mov    %r12,0x68(%rsp)
  40be5e:	4d 85 e4             	test   %r12,%r12
  40be61:	df 6c 24 68          	fildll 0x68(%rsp)
  40be65:	0f 88 85 07 00 00    	js     40c5f0 <__sprintf_chk@plt+0x9d60>
  40be6b:	48 89 5c 24 68       	mov    %rbx,0x68(%rsp)
  40be70:	48 85 db             	test   %rbx,%rbx
  40be73:	df 6c 24 68          	fildll 0x68(%rsp)
  40be77:	0f 88 8b 07 00 00    	js     40c608 <__sprintf_chk@plt+0x9d78>
  40be7d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40be82:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40be87:	48 85 c0             	test   %rax,%rax
  40be8a:	df 6c 24 68          	fildll 0x68(%rsp)
  40be8e:	0f 88 4c 07 00 00    	js     40c5e0 <__sprintf_chk@plt+0x9d50>
  40be94:	f6 44 24 20 10       	testb  $0x10,0x20(%rsp)
  40be99:	de f9                	fdivrp %st,%st(1)
  40be9b:	de c9                	fmulp  %st,%st(1)
  40be9d:	0f 84 15 03 00 00    	je     40c1b8 <__sprintf_chk@plt+0x9928>
  40bea3:	db 44 24 24          	fildl  0x24(%rsp)
  40bea7:	31 db                	xor    %ebx,%ebx
  40bea9:	d9 c0                	fld    %st(0)
  40beab:	eb 07                	jmp    40beb4 <__sprintf_chk@plt+0x9624>
  40bead:	0f 1f 00             	nopl   (%rax)
  40beb0:	dd d9                	fstp   %st(1)
  40beb2:	d9 ca                	fxch   %st(2)
  40beb4:	d9 c0                	fld    %st(0)
  40beb6:	83 c3 01             	add    $0x1,%ebx
  40beb9:	d8 ca                	fmul   %st(2),%st
  40bebb:	d9 cb                	fxch   %st(3)
  40bebd:	db eb                	fucomi %st(3),%st
  40bebf:	72 0f                	jb     40bed0 <__sprintf_chk@plt+0x9640>
  40bec1:	83 fb 08             	cmp    $0x8,%ebx
  40bec4:	75 ea                	jne    40beb0 <__sprintf_chk@plt+0x9620>
  40bec6:	dd da                	fstp   %st(2)
  40bec8:	dd da                	fstp   %st(2)
  40beca:	eb 08                	jmp    40bed4 <__sprintf_chk@plt+0x9644>
  40becc:	0f 1f 40 00          	nopl   0x0(%rax)
  40bed0:	dd da                	fstp   %st(2)
  40bed2:	dd da                	fstp   %st(2)
  40bed4:	de f1                	fdivp  %st,%st(1)
  40bed6:	83 7c 24 38 01       	cmpl   $0x1,0x38(%rsp)
  40bedb:	d9 c0                	fld    %st(0)
  40bedd:	0f 84 8d 00 00 00    	je     40bf70 <__sprintf_chk@plt+0x96e0>
  40bee3:	dd d8                	fstp   %st(0)
  40bee5:	db 2d a5 a0 00 00    	fldt   0xa0a5(%rip)        # 415f90 <_fini@@Base+0x4094>
  40beeb:	df e9                	fucomip %st(1),%st
  40beed:	0f 86 a5 05 00 00    	jbe    40c498 <__sprintf_chk@plt+0x9c08>
  40bef3:	d9 05 13 a0 00 00    	flds   0xa013(%rip)        # 415f0c <_fini@@Base+0x4010>
  40bef9:	d9 c9                	fxch   %st(1)
  40befb:	db e9                	fucomi %st(1),%st
  40befd:	0f 83 5d 07 00 00    	jae    40c660 <__sprintf_chk@plt+0x9dd0>
  40bf03:	dd d9                	fstp   %st(1)
  40bf05:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40bf09:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40bf0e:	80 cc 0c             	or     $0xc,%ah
  40bf11:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40bf16:	d9 c0                	fld    %st(0)
  40bf18:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40bf1c:	df 7c 24 68          	fistpll 0x68(%rsp)
  40bf20:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40bf24:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40bf29:	8b 4c 24 38          	mov    0x38(%rsp),%ecx
  40bf2d:	31 d2                	xor    %edx,%edx
  40bf2f:	85 c9                	test   %ecx,%ecx
  40bf31:	75 22                	jne    40bf55 <__sprintf_chk@plt+0x96c5>
  40bf33:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40bf38:	48 85 c0             	test   %rax,%rax
  40bf3b:	df 6c 24 68          	fildll 0x68(%rsp)
  40bf3f:	0f 88 38 08 00 00    	js     40c77d <__sprintf_chk@plt+0x9eed>
  40bf45:	31 c9                	xor    %ecx,%ecx
  40bf47:	ba 01 00 00 00       	mov    $0x1,%edx
  40bf4c:	df e9                	fucomip %st(1),%st
  40bf4e:	0f 9a c1             	setp   %cl
  40bf51:	48 0f 44 d1          	cmove  %rcx,%rdx
  40bf55:	48 01 d0             	add    %rdx,%rax
  40bf58:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40bf5d:	48 85 c0             	test   %rax,%rax
  40bf60:	df 6c 24 68          	fildll 0x68(%rsp)
  40bf64:	0f 88 ee 07 00 00    	js     40c758 <__sprintf_chk@plt+0x9ec8>
  40bf6a:	d9 c9                	fxch   %st(1)
  40bf6c:	eb 04                	jmp    40bf72 <__sprintf_chk@plt+0x96e2>
  40bf6e:	66 90                	xchg   %ax,%ax
  40bf70:	d9 c9                	fxch   %st(1)
  40bf72:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
  40bf77:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40bf7e:	b9 1e 5f 41 00       	mov    $0x415f1e,%ecx
  40bf83:	be 01 00 00 00       	mov    $0x1,%esi
  40bf88:	31 c0                	xor    %eax,%eax
  40bf8a:	4c 89 5c 24 50       	mov    %r11,0x50(%rsp)
  40bf8f:	4c 89 f7             	mov    %r14,%rdi
  40bf92:	db 7c 24 40          	fstpt  0x40(%rsp)
  40bf96:	db 3c 24             	fstpt  (%rsp)
  40bf99:	e8 f2 68 ff ff       	callq  402890 <__sprintf_chk@plt>
  40bf9e:	4c 89 f7             	mov    %r14,%rdi
  40bfa1:	e8 da 63 ff ff       	callq  402380 <strlen@plt>
  40bfa6:	4c 8b 5c 24 50       	mov    0x50(%rsp),%r11
  40bfab:	8b 54 24 58          	mov    0x58(%rsp),%edx
  40bfaf:	49 89 c7             	mov    %rax,%r15
  40bfb2:	31 c0                	xor    %eax,%eax
  40bfb4:	db 6c 24 40          	fldt   0x40(%rsp)
  40bfb8:	85 d2                	test   %edx,%edx
  40bfba:	4d 8d 73 01          	lea    0x1(%r11),%r14
  40bfbe:	0f 94 c0             	sete   %al
  40bfc1:	49 8d 44 06 01       	lea    0x1(%r14,%rax,1),%rax
  40bfc6:	49 39 c7             	cmp    %rax,%r15
  40bfc9:	77 25                	ja     40bff0 <__sprintf_chk@plt+0x9760>
  40bfcb:	f6 44 24 20 08       	testb  $0x8,0x20(%rsp)
  40bfd0:	0f 84 32 02 00 00    	je     40c208 <__sprintf_chk@plt+0x9978>
  40bfd6:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  40bfdb:	42 80 7c 38 ff 30    	cmpb   $0x30,-0x1(%rax,%r15,1)
  40bfe1:	0f 85 29 02 00 00    	jne    40c210 <__sprintf_chk@plt+0x9980>
  40bfe7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40bfee:	00 00 
  40bff0:	83 7c 24 38 01       	cmpl   $0x1,0x38(%rsp)
  40bff5:	d8 0d 89 9f 00 00    	fmuls  0x9f89(%rip)        # 415f84 <_fini@@Base+0x4088>
  40bffb:	0f 84 9f 00 00 00    	je     40c0a0 <__sprintf_chk@plt+0x9810>
  40c001:	db 2d 89 9f 00 00    	fldt   0x9f89(%rip)        # 415f90 <_fini@@Base+0x4094>
  40c007:	df e9                	fucomip %st(1),%st
  40c009:	0f 86 91 00 00 00    	jbe    40c0a0 <__sprintf_chk@plt+0x9810>
  40c00f:	d9 05 f7 9e 00 00    	flds   0x9ef7(%rip)        # 415f0c <_fini@@Base+0x4010>
  40c015:	d9 c9                	fxch   %st(1)
  40c017:	db e9                	fucomi %st(1),%st
  40c019:	0f 83 01 07 00 00    	jae    40c720 <__sprintf_chk@plt+0x9e90>
  40c01f:	dd d9                	fstp   %st(1)
  40c021:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40c025:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40c02a:	80 cc 0c             	or     $0xc,%ah
  40c02d:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40c032:	d9 c0                	fld    %st(0)
  40c034:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40c038:	df 7c 24 68          	fistpll 0x68(%rsp)
  40c03c:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40c040:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40c045:	44 8b 74 24 38       	mov    0x38(%rsp),%r14d
  40c04a:	31 d2                	xor    %edx,%edx
  40c04c:	45 85 f6             	test   %r14d,%r14d
  40c04f:	75 2f                	jne    40c080 <__sprintf_chk@plt+0x97f0>
  40c051:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40c056:	48 85 c0             	test   %rax,%rax
  40c059:	df 6c 24 68          	fildll 0x68(%rsp)
  40c05d:	0f 88 3b 07 00 00    	js     40c79e <__sprintf_chk@plt+0x9f0e>
  40c063:	31 c9                	xor    %ecx,%ecx
  40c065:	ba 01 00 00 00       	mov    $0x1,%edx
  40c06a:	df e9                	fucomip %st(1),%st
  40c06c:	dd d8                	fstp   %st(0)
  40c06e:	0f 9a c1             	setp   %cl
  40c071:	48 0f 44 d1          	cmove  %rcx,%rdx
  40c075:	eb 0b                	jmp    40c082 <__sprintf_chk@plt+0x97f2>
  40c077:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40c07e:	00 00 
  40c080:	dd d8                	fstp   %st(0)
  40c082:	48 01 d0             	add    %rdx,%rax
  40c085:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40c08a:	48 85 c0             	test   %rax,%rax
  40c08d:	df 6c 24 68          	fildll 0x68(%rsp)
  40c091:	79 0d                	jns    40c0a0 <__sprintf_chk@plt+0x9810>
  40c093:	d8 05 6f 9e 00 00    	fadds  0x9e6f(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c099:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40c0a0:	d8 35 de 9e 00 00    	fdivs  0x9ede(%rip)        # 415f84 <_fini@@Base+0x4088>
  40c0a6:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
  40c0ab:	b9 18 5f 41 00       	mov    $0x415f18,%ecx
  40c0b0:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40c0b7:	be 01 00 00 00       	mov    $0x1,%esi
  40c0bc:	31 c0                	xor    %eax,%eax
  40c0be:	4c 89 f7             	mov    %r14,%rdi
  40c0c1:	db 3c 24             	fstpt  (%rsp)
  40c0c4:	e8 c7 67 ff ff       	callq  402890 <__sprintf_chk@plt>
  40c0c9:	4c 89 f7             	mov    %r14,%rdi
  40c0cc:	45 31 f6             	xor    %r14d,%r14d
  40c0cf:	e8 ac 62 ff ff       	callq  402380 <strlen@plt>
  40c0d4:	49 89 c7             	mov    %rax,%r15
  40c0d7:	e9 3c 01 00 00       	jmpq   40c218 <__sprintf_chk@plt+0x9988>
  40c0dc:	0f 1f 40 00          	nopl   0x0(%rax)
  40c0e0:	48 85 db             	test   %rbx,%rbx
  40c0e3:	0f 84 70 fd ff ff    	je     40be59 <__sprintf_chk@plt+0x95c9>
  40c0e9:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40c0ee:	31 d2                	xor    %edx,%edx
  40c0f0:	48 f7 f3             	div    %rbx
  40c0f3:	48 85 d2             	test   %rdx,%rdx
  40c0f6:	48 89 c6             	mov    %rax,%rsi
  40c0f9:	0f 85 5a fd ff ff    	jne    40be59 <__sprintf_chk@plt+0x95c9>
  40c0ff:	31 d2                	xor    %edx,%edx
  40c101:	4c 89 e0             	mov    %r12,%rax
  40c104:	48 f7 f6             	div    %rsi
  40c107:	48 8d 3c 92          	lea    (%rdx,%rdx,4),%rdi
  40c10b:	49 89 c2             	mov    %rax,%r10
  40c10e:	31 d2                	xor    %edx,%edx
  40c110:	48 8d 04 3f          	lea    (%rdi,%rdi,1),%rax
  40c114:	48 f7 f6             	div    %rsi
  40c117:	48 01 d2             	add    %rdx,%rdx
  40c11a:	89 c7                	mov    %eax,%edi
  40c11c:	48 39 d6             	cmp    %rdx,%rsi
  40c11f:	0f 86 bb 05 00 00    	jbe    40c6e0 <__sprintf_chk@plt+0x9e50>
  40c125:	31 c9                	xor    %ecx,%ecx
  40c127:	48 85 d2             	test   %rdx,%rdx
  40c12a:	0f 95 c1             	setne  %cl
  40c12d:	44 8b 4c 24 20       	mov    0x20(%rsp),%r9d
  40c132:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40c137:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
  40c13c:	41 83 e1 10          	and    $0x10,%r9d
  40c140:	0f 84 ea 03 00 00    	je     40c530 <__sprintf_chk@plt+0x9ca0>
  40c146:	8b 74 24 24          	mov    0x24(%rsp),%esi
  40c14a:	49 39 f2             	cmp    %rsi,%r10
  40c14d:	0f 82 bd 05 00 00    	jb     40c710 <__sprintf_chk@plt+0x9e80>
  40c153:	31 db                	xor    %ebx,%ebx
  40c155:	44 8b 64 24 24       	mov    0x24(%rsp),%r12d
  40c15a:	eb 21                	jmp    40c17d <__sprintf_chk@plt+0x98ed>
  40c15c:	0f 1f 40 00          	nopl   0x0(%rax)
  40c160:	85 c9                	test   %ecx,%ecx
  40c162:	0f 95 c1             	setne  %cl
  40c165:	0f b6 c9             	movzbl %cl,%ecx
  40c168:	83 c3 01             	add    $0x1,%ebx
  40c16b:	4c 39 c6             	cmp    %r8,%rsi
  40c16e:	0f 87 2b 03 00 00    	ja     40c49f <__sprintf_chk@plt+0x9c0f>
  40c174:	83 fb 08             	cmp    $0x8,%ebx
  40c177:	0f 84 eb 05 00 00    	je     40c768 <__sprintf_chk@plt+0x9ed8>
  40c17d:	4c 89 d0             	mov    %r10,%rax
  40c180:	31 d2                	xor    %edx,%edx
  40c182:	48 f7 f6             	div    %rsi
  40c185:	49 89 c0             	mov    %rax,%r8
  40c188:	8d 04 92             	lea    (%rdx,%rdx,4),%eax
  40c18b:	31 d2                	xor    %edx,%edx
  40c18d:	4d 89 c2             	mov    %r8,%r10
  40c190:	8d 04 47             	lea    (%rdi,%rax,2),%eax
  40c193:	89 cf                	mov    %ecx,%edi
  40c195:	d1 ff                	sar    %edi
  40c197:	41 f7 f4             	div    %r12d
  40c19a:	44 8d 34 57          	lea    (%rdi,%rdx,2),%r14d
  40c19e:	89 c7                	mov    %eax,%edi
  40c1a0:	44 01 f1             	add    %r14d,%ecx
  40c1a3:	45 39 f4             	cmp    %r14d,%r12d
  40c1a6:	77 b8                	ja     40c160 <__sprintf_chk@plt+0x98d0>
  40c1a8:	41 39 cc             	cmp    %ecx,%r12d
  40c1ab:	19 c9                	sbb    %ecx,%ecx
  40c1ad:	f7 d1                	not    %ecx
  40c1af:	83 c1 03             	add    $0x3,%ecx
  40c1b2:	eb b4                	jmp    40c168 <__sprintf_chk@plt+0x98d8>
  40c1b4:	0f 1f 40 00          	nopl   0x0(%rax)
  40c1b8:	83 7c 24 38 01       	cmpl   $0x1,0x38(%rsp)
  40c1bd:	74 0e                	je     40c1cd <__sprintf_chk@plt+0x993d>
  40c1bf:	db 2d cb 9d 00 00    	fldt   0x9dcb(%rip)        # 415f90 <_fini@@Base+0x4094>
  40c1c5:	df e9                	fucomip %st(1),%st
  40c1c7:	0f 87 3b 02 00 00    	ja     40c408 <__sprintf_chk@plt+0x9b78>
  40c1cd:	db 3c 24             	fstpt  (%rsp)
  40c1d0:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  40c1d5:	b9 18 5f 41 00       	mov    $0x415f18,%ecx
  40c1da:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40c1e1:	be 01 00 00 00       	mov    $0x1,%esi
  40c1e6:	31 c0                	xor    %eax,%eax
  40c1e8:	45 31 f6             	xor    %r14d,%r14d
  40c1eb:	48 89 df             	mov    %rbx,%rdi
  40c1ee:	e8 9d 66 ff ff       	callq  402890 <__sprintf_chk@plt>
  40c1f3:	48 89 df             	mov    %rbx,%rdi
  40c1f6:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
  40c1fb:	e8 80 61 ff ff       	callq  402380 <strlen@plt>
  40c200:	49 89 c7             	mov    %rax,%r15
  40c203:	eb 13                	jmp    40c218 <__sprintf_chk@plt+0x9988>
  40c205:	0f 1f 00             	nopl   (%rax)
  40c208:	dd d8                	fstp   %st(0)
  40c20a:	eb 0c                	jmp    40c218 <__sprintf_chk@plt+0x9988>
  40c20c:	0f 1f 40 00          	nopl   0x0(%rax)
  40c210:	dd d8                	fstp   %st(0)
  40c212:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40c218:	4c 8b 64 24 18       	mov    0x18(%rsp),%r12
  40c21d:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
  40c222:	4c 89 fa             	mov    %r15,%rdx
  40c225:	4d 29 fc             	sub    %r15,%r12
  40c228:	4d 29 f7             	sub    %r14,%r15
  40c22b:	4c 89 e7             	mov    %r12,%rdi
  40c22e:	e8 2d 65 ff ff       	callq  402760 <memmove@plt>
  40c233:	4f 8d 04 3c          	lea    (%r12,%r15,1),%r8
  40c237:	f6 44 24 20 04       	testb  $0x4,0x20(%rsp)
  40c23c:	0f 85 d6 00 00 00    	jne    40c318 <__sprintf_chk@plt+0x9a88>
  40c242:	f6 44 24 20 80       	testb  $0x80,0x20(%rsp)
  40c247:	74 71                	je     40c2ba <__sprintf_chk@plt+0x9a2a>
  40c249:	83 fb ff             	cmp    $0xffffffff,%ebx
  40c24c:	0f 84 c6 03 00 00    	je     40c618 <__sprintf_chk@plt+0x9d88>
  40c252:	8b 44 24 20          	mov    0x20(%rsp),%eax
  40c256:	89 d9                	mov    %ebx,%ecx
  40c258:	25 00 01 00 00       	and    $0x100,%eax
  40c25d:	09 c1                	or     %eax,%ecx
  40c25f:	74 59                	je     40c2ba <__sprintf_chk@plt+0x9a2a>
  40c261:	f6 44 24 20 40       	testb  $0x40,0x20(%rsp)
  40c266:	0f 85 7c 01 00 00    	jne    40c3e8 <__sprintf_chk@plt+0x9b58>
  40c26c:	85 db                	test   %ebx,%ebx
  40c26e:	0f 84 77 05 00 00    	je     40c7eb <__sprintf_chk@plt+0x9f5b>
  40c274:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40c279:	48 8d 51 01          	lea    0x1(%rcx),%rdx
  40c27d:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
  40c281:	85 c9                	test   %ecx,%ecx
  40c283:	0f 84 47 01 00 00    	je     40c3d0 <__sprintf_chk@plt+0x9b40>
  40c289:	48 63 cb             	movslq %ebx,%rcx
  40c28c:	0f b6 89 78 5f 41 00 	movzbl 0x415f78(%rcx),%ecx
  40c293:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  40c298:	88 0f                	mov    %cl,(%rdi)
  40c29a:	85 c0                	test   %eax,%eax
  40c29c:	0f 84 53 05 00 00    	je     40c7f5 <__sprintf_chk@plt+0x9f65>
  40c2a2:	8b 44 24 58          	mov    0x58(%rsp),%eax
  40c2a6:	85 c0                	test   %eax,%eax
  40c2a8:	0f 85 0a 01 00 00    	jne    40c3b8 <__sprintf_chk@plt+0x9b28>
  40c2ae:	48 8d 42 01          	lea    0x1(%rdx),%rax
  40c2b2:	c6 02 42             	movb   $0x42,(%rdx)
  40c2b5:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40c2ba:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40c2bf:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  40c2c6:	00 
  40c2c7:	64 48 33 3c 25 28 00 	xor    %fs:0x28,%rdi
  40c2ce:	00 00 
  40c2d0:	c6 00 00             	movb   $0x0,(%rax)
  40c2d3:	4c 89 e0             	mov    %r12,%rax
  40c2d6:	0f 85 0a 05 00 00    	jne    40c7e6 <__sprintf_chk@plt+0x9f56>
  40c2dc:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
  40c2e3:	5b                   	pop    %rbx
  40c2e4:	5d                   	pop    %rbp
  40c2e5:	41 5c                	pop    %r12
  40c2e7:	41 5d                	pop    %r13
  40c2e9:	41 5e                	pop    %r14
  40c2eb:	41 5f                	pop    %r15
  40c2ed:	c3                   	retq   
  40c2ee:	66 90                	xchg   %ax,%ax
  40c2f0:	49 89 c2             	mov    %rax,%r10
  40c2f3:	31 d2                	xor    %edx,%edx
  40c2f5:	4d 0f af d4          	imul   %r12,%r10
  40c2f9:	4c 89 d0             	mov    %r10,%rax
  40c2fc:	48 f7 f1             	div    %rcx
  40c2ff:	4c 39 e0             	cmp    %r12,%rax
  40c302:	0f 85 51 fb ff ff    	jne    40be59 <__sprintf_chk@plt+0x95c9>
  40c308:	31 c9                	xor    %ecx,%ecx
  40c30a:	31 ff                	xor    %edi,%edi
  40c30c:	e9 1c fe ff ff       	jmpq   40c12d <__sprintf_chk@plt+0x989d>
  40c311:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40c318:	4d 29 e0             	sub    %r12,%r8
  40c31b:	4c 89 ef             	mov    %r13,%rdi
  40c31e:	49 c7 c7 ff ff ff ff 	mov    $0xffffffffffffffff,%r15
  40c325:	4d 89 c6             	mov    %r8,%r14
  40c328:	e8 53 60 ff ff       	callq  402380 <strlen@plt>
  40c32d:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40c332:	4c 89 e6             	mov    %r12,%rsi
  40c335:	b9 29 00 00 00       	mov    $0x29,%ecx
  40c33a:	4c 89 f2             	mov    %r14,%rdx
  40c33d:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  40c342:	e8 49 62 ff ff       	callq  402590 <__memcpy_chk@plt>
  40c347:	4f 8d 0c 34          	lea    (%r12,%r14,1),%r9
  40c34b:	4d 89 ec             	mov    %r13,%r12
  40c34e:	4c 8b 6c 24 38       	mov    0x38(%rsp),%r13
  40c353:	eb 17                	jmp    40c36c <__sprintf_chk@plt+0x9adc>
  40c355:	0f 1f 00             	nopl   (%rax)
  40c358:	4d 29 e9             	sub    %r13,%r9
  40c35b:	4c 89 ea             	mov    %r13,%rdx
  40c35e:	4c 89 e6             	mov    %r12,%rsi
  40c361:	4c 89 cf             	mov    %r9,%rdi
  40c364:	e8 57 62 ff ff       	callq  4025c0 <memcpy@plt>
  40c369:	49 89 c1             	mov    %rax,%r9
  40c36c:	0f b6 55 00          	movzbl 0x0(%rbp),%edx
  40c370:	84 d2                	test   %dl,%dl
  40c372:	74 0f                	je     40c383 <__sprintf_chk@plt+0x9af3>
  40c374:	80 fa 7e             	cmp    $0x7e,%dl
  40c377:	44 0f b6 fa          	movzbl %dl,%r15d
  40c37b:	4d 0f 47 fe          	cmova  %r14,%r15
  40c37f:	48 83 c5 01          	add    $0x1,%rbp
  40c383:	4d 39 f7             	cmp    %r14,%r15
  40c386:	48 8d 44 24 70       	lea    0x70(%rsp),%rax
  40c38b:	4d 0f 47 fe          	cmova  %r14,%r15
  40c38f:	4d 29 fe             	sub    %r15,%r14
  40c392:	4d 29 f9             	sub    %r15,%r9
  40c395:	4c 89 fa             	mov    %r15,%rdx
  40c398:	4a 8d 34 30          	lea    (%rax,%r14,1),%rsi
  40c39c:	4c 89 cf             	mov    %r9,%rdi
  40c39f:	e8 1c 62 ff ff       	callq  4025c0 <memcpy@plt>
  40c3a4:	4d 85 f6             	test   %r14,%r14
  40c3a7:	49 89 c1             	mov    %rax,%r9
  40c3aa:	75 ac                	jne    40c358 <__sprintf_chk@plt+0x9ac8>
  40c3ac:	49 89 c4             	mov    %rax,%r12
  40c3af:	e9 8e fe ff ff       	jmpq   40c242 <__sprintf_chk@plt+0x99b2>
  40c3b4:	0f 1f 40 00          	nopl   0x0(%rax)
  40c3b8:	85 db                	test   %ebx,%ebx
  40c3ba:	0f 84 ee fe ff ff    	je     40c2ae <__sprintf_chk@plt+0x9a1e>
  40c3c0:	c6 02 69             	movb   $0x69,(%rdx)
  40c3c3:	48 83 c2 01          	add    $0x1,%rdx
  40c3c7:	e9 e2 fe ff ff       	jmpq   40c2ae <__sprintf_chk@plt+0x9a1e>
  40c3cc:	0f 1f 40 00          	nopl   0x0(%rax)
  40c3d0:	83 fb 01             	cmp    $0x1,%ebx
  40c3d3:	b9 6b 00 00 00       	mov    $0x6b,%ecx
  40c3d8:	0f 85 ab fe ff ff    	jne    40c289 <__sprintf_chk@plt+0x99f9>
  40c3de:	e9 b0 fe ff ff       	jmpq   40c293 <__sprintf_chk@plt+0x9a03>
  40c3e3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40c3e8:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  40c3ed:	48 8d b1 89 02 00 00 	lea    0x289(%rcx),%rsi
  40c3f4:	c6 81 88 02 00 00 20 	movb   $0x20,0x288(%rcx)
  40c3fb:	48 89 74 24 18       	mov    %rsi,0x18(%rsp)
  40c400:	e9 67 fe ff ff       	jmpq   40c26c <__sprintf_chk@plt+0x99dc>
  40c405:	0f 1f 00             	nopl   (%rax)
  40c408:	d9 05 fe 9a 00 00    	flds   0x9afe(%rip)        # 415f0c <_fini@@Base+0x4010>
  40c40e:	d9 c9                	fxch   %st(1)
  40c410:	db e9                	fucomi %st(1),%st
  40c412:	0f 83 88 02 00 00    	jae    40c6a0 <__sprintf_chk@plt+0x9e10>
  40c418:	dd d9                	fstp   %st(1)
  40c41a:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40c41e:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40c423:	80 cc 0c             	or     $0xc,%ah
  40c426:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40c42b:	d9 c0                	fld    %st(0)
  40c42d:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40c431:	df 7c 24 68          	fistpll 0x68(%rsp)
  40c435:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40c439:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40c43e:	8b 74 24 38          	mov    0x38(%rsp),%esi
  40c442:	31 d2                	xor    %edx,%edx
  40c444:	85 f6                	test   %esi,%esi
  40c446:	75 28                	jne    40c470 <__sprintf_chk@plt+0x9be0>
  40c448:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40c44d:	48 85 c0             	test   %rax,%rax
  40c450:	df 6c 24 68          	fildll 0x68(%rsp)
  40c454:	0f 88 18 03 00 00    	js     40c772 <__sprintf_chk@plt+0x9ee2>
  40c45a:	31 c9                	xor    %ecx,%ecx
  40c45c:	ba 01 00 00 00       	mov    $0x1,%edx
  40c461:	df e9                	fucomip %st(1),%st
  40c463:	dd d8                	fstp   %st(0)
  40c465:	0f 9a c1             	setp   %cl
  40c468:	48 0f 44 d1          	cmove  %rcx,%rdx
  40c46c:	eb 04                	jmp    40c472 <__sprintf_chk@plt+0x9be2>
  40c46e:	66 90                	xchg   %ax,%ax
  40c470:	dd d8                	fstp   %st(0)
  40c472:	48 01 d0             	add    %rdx,%rax
  40c475:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40c47a:	48 85 c0             	test   %rax,%rax
  40c47d:	df 6c 24 68          	fildll 0x68(%rsp)
  40c481:	0f 89 46 fd ff ff    	jns    40c1cd <__sprintf_chk@plt+0x993d>
  40c487:	d8 05 7b 9a 00 00    	fadds  0x9a7b(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c48d:	e9 3b fd ff ff       	jmpq   40c1cd <__sprintf_chk@plt+0x993d>
  40c492:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40c498:	d9 c0                	fld    %st(0)
  40c49a:	e9 d3 fa ff ff       	jmpq   40bf72 <__sprintf_chk@plt+0x96e2>
  40c49f:	49 83 f8 09          	cmp    $0x9,%r8
  40c4a3:	0f 87 bf 02 00 00    	ja     40c768 <__sprintf_chk@plt+0x9ed8>
  40c4a9:	83 7c 24 38 01       	cmpl   $0x1,0x38(%rsp)
  40c4ae:	0f 84 fc 02 00 00    	je     40c7b0 <__sprintf_chk@plt+0x9f20>
  40c4b4:	44 8b 64 24 38       	mov    0x38(%rsp),%r12d
  40c4b9:	85 c9                	test   %ecx,%ecx
  40c4bb:	0f 9f c2             	setg   %dl
  40c4be:	45 85 e4             	test   %r12d,%r12d
  40c4c1:	40 0f 94 c6          	sete   %sil
  40c4c5:	21 f2                	and    %esi,%edx
  40c4c7:	84 d2                	test   %dl,%dl
  40c4c9:	0f 84 b9 02 00 00    	je     40c788 <__sprintf_chk@plt+0x9ef8>
  40c4cf:	8d 78 01             	lea    0x1(%rax),%edi
  40c4d2:	83 ff 0a             	cmp    $0xa,%edi
  40c4d5:	0f 84 f3 02 00 00    	je     40c7ce <__sprintf_chk@plt+0x9f3e>
  40c4db:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  40c4e0:	83 c7 30             	add    $0x30,%edi
  40c4e3:	4c 89 da             	mov    %r11,%rdx
  40c4e6:	4c 89 fe             	mov    %r15,%rsi
  40c4e9:	44 89 4c 24 5c       	mov    %r9d,0x5c(%rsp)
  40c4ee:	4c 89 54 24 40       	mov    %r10,0x40(%rsp)
  40c4f3:	4c 89 5c 24 50       	mov    %r11,0x50(%rsp)
  40c4f8:	4c 8d 80 87 02 00 00 	lea    0x287(%rax),%r8
  40c4ff:	40 88 b8 87 02 00 00 	mov    %dil,0x287(%rax)
  40c506:	4d 29 d8             	sub    %r11,%r8
  40c509:	4c 89 c7             	mov    %r8,%rdi
  40c50c:	e8 af 60 ff ff       	callq  4025c0 <memcpy@plt>
  40c511:	4c 8b 5c 24 50       	mov    0x50(%rsp),%r11
  40c516:	4c 8b 54 24 40       	mov    0x40(%rsp),%r10
  40c51b:	49 89 c0             	mov    %rax,%r8
  40c51e:	44 8b 4c 24 5c       	mov    0x5c(%rsp),%r9d
  40c523:	31 c9                	xor    %ecx,%ecx
  40c525:	31 ff                	xor    %edi,%edi
  40c527:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40c52e:	00 00 
  40c530:	83 7c 24 38 01       	cmpl   $0x1,0x38(%rsp)
  40c535:	0f 84 b5 01 00 00    	je     40c6f0 <__sprintf_chk@plt+0x9e60>
  40c53b:	8b 74 24 38          	mov    0x38(%rsp),%esi
  40c53f:	31 c0                	xor    %eax,%eax
  40c541:	85 f6                	test   %esi,%esi
  40c543:	75 07                	jne    40c54c <__sprintf_chk@plt+0x9cbc>
  40c545:	01 f9                	add    %edi,%ecx
  40c547:	85 c9                	test   %ecx,%ecx
  40c549:	0f 9f c0             	setg   %al
  40c54c:	84 c0                	test   %al,%al
  40c54e:	74 50                	je     40c5a0 <__sprintf_chk@plt+0x9d10>
  40c550:	49 83 c2 01          	add    $0x1,%r10
  40c554:	45 85 c9             	test   %r9d,%r9d
  40c557:	74 47                	je     40c5a0 <__sprintf_chk@plt+0x9d10>
  40c559:	8b 44 24 24          	mov    0x24(%rsp),%eax
  40c55d:	4c 39 d0             	cmp    %r10,%rax
  40c560:	75 3e                	jne    40c5a0 <__sprintf_chk@plt+0x9d10>
  40c562:	83 fb 08             	cmp    $0x8,%ebx
  40c565:	74 39                	je     40c5a0 <__sprintf_chk@plt+0x9d10>
  40c567:	83 c3 01             	add    $0x1,%ebx
  40c56a:	f6 44 24 20 08       	testb  $0x8,0x20(%rsp)
  40c56f:	41 ba 01 00 00 00    	mov    $0x1,%r10d
  40c575:	75 29                	jne    40c5a0 <__sprintf_chk@plt+0x9d10>
  40c577:	49 8d 40 ff          	lea    -0x1(%r8),%rax
  40c57b:	41 c6 40 ff 30       	movb   $0x30,-0x1(%r8)
  40c580:	4c 89 da             	mov    %r11,%rdx
  40c583:	4c 89 fe             	mov    %r15,%rsi
  40c586:	4c 89 54 24 38       	mov    %r10,0x38(%rsp)
  40c58b:	4c 29 d8             	sub    %r11,%rax
  40c58e:	48 89 c7             	mov    %rax,%rdi
  40c591:	e8 2a 60 ff ff       	callq  4025c0 <memcpy@plt>
  40c596:	4c 8b 54 24 38       	mov    0x38(%rsp),%r10
  40c59b:	49 89 c0             	mov    %rax,%r8
  40c59e:	66 90                	xchg   %ax,%ax
  40c5a0:	4d 89 c4             	mov    %r8,%r12
  40c5a3:	48 b9 cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rcx
  40c5aa:	cc cc cc 
  40c5ad:	0f 1f 00             	nopl   (%rax)
  40c5b0:	4c 89 d0             	mov    %r10,%rax
  40c5b3:	49 83 ec 01          	sub    $0x1,%r12
  40c5b7:	48 f7 e1             	mul    %rcx
  40c5ba:	48 c1 ea 03          	shr    $0x3,%rdx
  40c5be:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  40c5c2:	48 01 c0             	add    %rax,%rax
  40c5c5:	49 29 c2             	sub    %rax,%r10
  40c5c8:	41 83 c2 30          	add    $0x30,%r10d
  40c5cc:	48 85 d2             	test   %rdx,%rdx
  40c5cf:	45 88 14 24          	mov    %r10b,(%r12)
  40c5d3:	49 89 d2             	mov    %rdx,%r10
  40c5d6:	75 d8                	jne    40c5b0 <__sprintf_chk@plt+0x9d20>
  40c5d8:	e9 5a fc ff ff       	jmpq   40c237 <__sprintf_chk@plt+0x99a7>
  40c5dd:	0f 1f 00             	nopl   (%rax)
  40c5e0:	d8 05 22 99 00 00    	fadds  0x9922(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c5e6:	e9 a9 f8 ff ff       	jmpq   40be94 <__sprintf_chk@plt+0x9604>
  40c5eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40c5f0:	d8 05 12 99 00 00    	fadds  0x9912(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c5f6:	48 89 5c 24 68       	mov    %rbx,0x68(%rsp)
  40c5fb:	48 85 db             	test   %rbx,%rbx
  40c5fe:	df 6c 24 68          	fildll 0x68(%rsp)
  40c602:	0f 89 75 f8 ff ff    	jns    40be7d <__sprintf_chk@plt+0x95ed>
  40c608:	d8 05 fa 98 00 00    	fadds  0x98fa(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c60e:	e9 6a f8 ff ff       	jmpq   40be7d <__sprintf_chk@plt+0x95ed>
  40c613:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40c618:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  40c61d:	48 83 fa 01          	cmp    $0x1,%rdx
  40c621:	0f 86 82 01 00 00    	jbe    40c7a9 <__sprintf_chk@plt+0x9f19>
  40c627:	44 8b 74 24 24       	mov    0x24(%rsp),%r14d
  40c62c:	bb 01 00 00 00       	mov    $0x1,%ebx
  40c631:	b8 01 00 00 00       	mov    $0x1,%eax
  40c636:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40c63d:	00 00 00 
  40c640:	49 0f af c6          	imul   %r14,%rax
  40c644:	48 39 c2             	cmp    %rax,%rdx
  40c647:	0f 86 05 fc ff ff    	jbe    40c252 <__sprintf_chk@plt+0x99c2>
  40c64d:	83 c3 01             	add    $0x1,%ebx
  40c650:	83 fb 08             	cmp    $0x8,%ebx
  40c653:	75 eb                	jne    40c640 <__sprintf_chk@plt+0x9db0>
  40c655:	e9 f8 fb ff ff       	jmpq   40c252 <__sprintf_chk@plt+0x99c2>
  40c65a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40c660:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40c664:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40c669:	dc e1                	fsub   %st,%st(1)
  40c66b:	d9 c9                	fxch   %st(1)
  40c66d:	48 ba 00 00 00 00 00 	movabs $0x8000000000000000,%rdx
  40c674:	00 00 80 
  40c677:	80 cc 0c             	or     $0xc,%ah
  40c67a:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40c67f:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40c683:	df 7c 24 68          	fistpll 0x68(%rsp)
  40c687:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40c68b:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40c690:	48 31 d0             	xor    %rdx,%rax
  40c693:	e9 91 f8 ff ff       	jmpq   40bf29 <__sprintf_chk@plt+0x9699>
  40c698:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40c69f:	00 
  40c6a0:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40c6a4:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40c6a9:	dc e1                	fsub   %st,%st(1)
  40c6ab:	d9 c9                	fxch   %st(1)
  40c6ad:	48 ba 00 00 00 00 00 	movabs $0x8000000000000000,%rdx
  40c6b4:	00 00 80 
  40c6b7:	80 cc 0c             	or     $0xc,%ah
  40c6ba:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40c6bf:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40c6c3:	df 7c 24 68          	fistpll 0x68(%rsp)
  40c6c7:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40c6cb:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40c6d0:	48 31 d0             	xor    %rdx,%rax
  40c6d3:	e9 66 fd ff ff       	jmpq   40c43e <__sprintf_chk@plt+0x9bae>
  40c6d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40c6df:	00 
  40c6e0:	19 c9                	sbb    %ecx,%ecx
  40c6e2:	f7 d1                	not    %ecx
  40c6e4:	83 c1 03             	add    $0x3,%ecx
  40c6e7:	e9 41 fa ff ff       	jmpq   40c12d <__sprintf_chk@plt+0x989d>
  40c6ec:	0f 1f 40 00          	nopl   0x0(%rax)
  40c6f0:	4c 89 d0             	mov    %r10,%rax
  40c6f3:	48 63 c9             	movslq %ecx,%rcx
  40c6f6:	83 e0 01             	and    $0x1,%eax
  40c6f9:	48 01 c8             	add    %rcx,%rax
  40c6fc:	0f 95 c0             	setne  %al
  40c6ff:	0f b6 c0             	movzbl %al,%eax
  40c702:	01 c7                	add    %eax,%edi
  40c704:	83 ff 05             	cmp    $0x5,%edi
  40c707:	0f 9f c0             	setg   %al
  40c70a:	e9 3d fe ff ff       	jmpq   40c54c <__sprintf_chk@plt+0x9cbc>
  40c70f:	90                   	nop
  40c710:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40c715:	31 db                	xor    %ebx,%ebx
  40c717:	e9 14 fe ff ff       	jmpq   40c530 <__sprintf_chk@plt+0x9ca0>
  40c71c:	0f 1f 40 00          	nopl   0x0(%rax)
  40c720:	d9 7c 24 66          	fnstcw 0x66(%rsp)
  40c724:	0f b7 44 24 66       	movzwl 0x66(%rsp),%eax
  40c729:	dc e1                	fsub   %st,%st(1)
  40c72b:	d9 c9                	fxch   %st(1)
  40c72d:	48 ba 00 00 00 00 00 	movabs $0x8000000000000000,%rdx
  40c734:	00 00 80 
  40c737:	80 cc 0c             	or     $0xc,%ah
  40c73a:	66 89 44 24 64       	mov    %ax,0x64(%rsp)
  40c73f:	d9 6c 24 64          	fldcw  0x64(%rsp)
  40c743:	df 7c 24 68          	fistpll 0x68(%rsp)
  40c747:	d9 6c 24 66          	fldcw  0x66(%rsp)
  40c74b:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40c750:	48 31 d0             	xor    %rdx,%rax
  40c753:	e9 ed f8 ff ff       	jmpq   40c045 <__sprintf_chk@plt+0x97b5>
  40c758:	d8 05 aa 97 00 00    	fadds  0x97aa(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c75e:	d9 c9                	fxch   %st(1)
  40c760:	e9 0d f8 ff ff       	jmpq   40bf72 <__sprintf_chk@plt+0x96e2>
  40c765:	0f 1f 00             	nopl   (%rax)
  40c768:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40c76d:	e9 be fd ff ff       	jmpq   40c530 <__sprintf_chk@plt+0x9ca0>
  40c772:	d8 05 90 97 00 00    	fadds  0x9790(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c778:	e9 dd fc ff ff       	jmpq   40c45a <__sprintf_chk@plt+0x9bca>
  40c77d:	d8 05 85 97 00 00    	fadds  0x9785(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c783:	e9 bd f7 ff ff       	jmpq   40bf45 <__sprintf_chk@plt+0x96b5>
  40c788:	85 c0                	test   %eax,%eax
  40c78a:	0f 85 4b fd ff ff    	jne    40c4db <__sprintf_chk@plt+0x9c4b>
  40c790:	f6 44 24 20 08       	testb  $0x8,0x20(%rsp)
  40c795:	75 2b                	jne    40c7c2 <__sprintf_chk@plt+0x9f32>
  40c797:	31 ff                	xor    %edi,%edi
  40c799:	e9 3d fd ff ff       	jmpq   40c4db <__sprintf_chk@plt+0x9c4b>
  40c79e:	d8 05 64 97 00 00    	fadds  0x9764(%rip)        # 415f08 <_fini@@Base+0x400c>
  40c7a4:	e9 ba f8 ff ff       	jmpq   40c063 <__sprintf_chk@plt+0x97d3>
  40c7a9:	31 db                	xor    %ebx,%ebx
  40c7ab:	e9 a2 fa ff ff       	jmpq   40c252 <__sprintf_chk@plt+0x99c2>
  40c7b0:	89 c2                	mov    %eax,%edx
  40c7b2:	83 e2 01             	and    $0x1,%edx
  40c7b5:	01 ca                	add    %ecx,%edx
  40c7b7:	83 fa 02             	cmp    $0x2,%edx
  40c7ba:	0f 9f c2             	setg   %dl
  40c7bd:	e9 05 fd ff ff       	jmpq   40c4c7 <__sprintf_chk@plt+0x9c37>
  40c7c2:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40c7c7:	31 ff                	xor    %edi,%edi
  40c7c9:	e9 62 fd ff ff       	jmpq   40c530 <__sprintf_chk@plt+0x9ca0>
  40c7ce:	4d 8d 50 01          	lea    0x1(%r8),%r10
  40c7d2:	49 83 fa 0a          	cmp    $0xa,%r10
  40c7d6:	75 27                	jne    40c7ff <__sprintf_chk@plt+0x9f6f>
  40c7d8:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40c7dd:	31 c9                	xor    %ecx,%ecx
  40c7df:	31 ff                	xor    %edi,%edi
  40c7e1:	e9 4a fd ff ff       	jmpq   40c530 <__sprintf_chk@plt+0x9ca0>
  40c7e6:	e8 b5 5b ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  40c7eb:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  40c7f0:	e9 a5 fa ff ff       	jmpq   40c29a <__sprintf_chk@plt+0x9a0a>
  40c7f5:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  40c7fa:	e9 bb fa ff ff       	jmpq   40c2ba <__sprintf_chk@plt+0x9a2a>
  40c7ff:	31 c9                	xor    %ecx,%ecx
  40c801:	eb 8d                	jmp    40c790 <__sprintf_chk@plt+0x9f00>
  40c803:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40c80a:	84 00 00 00 00 00 
  40c810:	41 55                	push   %r13
  40c812:	49 89 f5             	mov    %rsi,%r13
  40c815:	41 54                	push   %r12
  40c817:	49 89 d4             	mov    %rdx,%r12
  40c81a:	55                   	push   %rbp
  40c81b:	53                   	push   %rbx
  40c81c:	48 89 fb             	mov    %rdi,%rbx
  40c81f:	48 83 ec 18          	sub    $0x18,%rsp
  40c823:	48 85 ff             	test   %rdi,%rdi
  40c826:	0f 84 e4 00 00 00    	je     40c910 <__sprintf_chk@plt+0xa080>
  40c82c:	31 ed                	xor    %ebp,%ebp
  40c82e:	80 3b 27             	cmpb   $0x27,(%rbx)
  40c831:	74 6d                	je     40c8a0 <__sprintf_chk@plt+0xa010>
  40c833:	b9 04 00 00 00       	mov    $0x4,%ecx
  40c838:	ba 50 5f 41 00       	mov    $0x415f50,%edx
  40c83d:	be 60 5f 41 00       	mov    $0x415f60,%esi
  40c842:	48 89 df             	mov    %rbx,%rdi
  40c845:	e8 06 d6 ff ff       	callq  409e50 <__sprintf_chk@plt+0x75c0>
  40c84a:	85 c0                	test   %eax,%eax
  40c84c:	78 62                	js     40c8b0 <__sprintf_chk@plt+0xa020>
  40c84e:	48 98                	cltq   
  40c850:	49 c7 04 24 01 00 00 	movq   $0x1,(%r12)
  40c857:	00 
  40c858:	ba 01 00 00 00       	mov    $0x1,%edx
  40c85d:	0b 2c 85 50 5f 41 00 	or     0x415f50(,%rax,4),%ebp
  40c864:	41 89 6d 00          	mov    %ebp,0x0(%r13)
  40c868:	31 c0                	xor    %eax,%eax
  40c86a:	48 85 d2             	test   %rdx,%rdx
  40c86d:	75 25                	jne    40c894 <__sprintf_chk@plt+0xa004>
  40c86f:	bf 2e 5f 41 00       	mov    $0x415f2e,%edi
  40c874:	e8 47 59 ff ff       	callq  4021c0 <getenv@plt>
  40c879:	48 83 f8 01          	cmp    $0x1,%rax
  40c87d:	48 19 c0             	sbb    %rax,%rax
  40c880:	25 00 02 00 00       	and    $0x200,%eax
  40c885:	48 05 00 02 00 00    	add    $0x200,%rax
  40c88b:	49 89 04 24          	mov    %rax,(%r12)
  40c88f:	b8 04 00 00 00       	mov    $0x4,%eax
  40c894:	48 83 c4 18          	add    $0x18,%rsp
  40c898:	5b                   	pop    %rbx
  40c899:	5d                   	pop    %rbp
  40c89a:	41 5c                	pop    %r12
  40c89c:	41 5d                	pop    %r13
  40c89e:	c3                   	retq   
  40c89f:	90                   	nop
  40c8a0:	48 83 c3 01          	add    $0x1,%rbx
  40c8a4:	40 b5 04             	mov    $0x4,%bpl
  40c8a7:	eb 8a                	jmp    40c833 <__sprintf_chk@plt+0x9fa3>
  40c8a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40c8b0:	48 8d 74 24 08       	lea    0x8(%rsp),%rsi
  40c8b5:	31 d2                	xor    %edx,%edx
  40c8b7:	41 b8 3e 5f 41 00    	mov    $0x415f3e,%r8d
  40c8bd:	4c 89 e1             	mov    %r12,%rcx
  40c8c0:	48 89 df             	mov    %rbx,%rdi
  40c8c3:	e8 98 4a 00 00       	callq  411360 <__sprintf_chk@plt+0xead0>
  40c8c8:	85 c0                	test   %eax,%eax
  40c8ca:	0f 85 98 00 00 00    	jne    40c968 <__sprintf_chk@plt+0xa0d8>
  40c8d0:	0f b6 03             	movzbl (%rbx),%eax
  40c8d3:	83 e8 30             	sub    $0x30,%eax
  40c8d6:	3c 09                	cmp    $0x9,%al
  40c8d8:	76 2d                	jbe    40c907 <__sprintf_chk@plt+0xa077>
  40c8da:	48 8b 54 24 08       	mov    0x8(%rsp),%rdx
  40c8df:	48 39 d3             	cmp    %rdx,%rbx
  40c8e2:	75 15                	jne    40c8f9 <__sprintf_chk@plt+0xa069>
  40c8e4:	e9 97 00 00 00       	jmpq   40c980 <__sprintf_chk@plt+0xa0f0>
  40c8e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40c8f0:	48 39 d3             	cmp    %rdx,%rbx
  40c8f3:	0f 84 87 00 00 00    	je     40c980 <__sprintf_chk@plt+0xa0f0>
  40c8f9:	48 83 c3 01          	add    $0x1,%rbx
  40c8fd:	0f b6 03             	movzbl (%rbx),%eax
  40c900:	83 e8 30             	sub    $0x30,%eax
  40c903:	3c 09                	cmp    $0x9,%al
  40c905:	77 e9                	ja     40c8f0 <__sprintf_chk@plt+0xa060>
  40c907:	49 8b 14 24          	mov    (%r12),%rdx
  40c90b:	e9 54 ff ff ff       	jmpq   40c864 <__sprintf_chk@plt+0x9fd4>
  40c910:	bf e4 38 41 00       	mov    $0x4138e4,%edi
  40c915:	e8 a6 58 ff ff       	callq  4021c0 <getenv@plt>
  40c91a:	48 85 c0             	test   %rax,%rax
  40c91d:	48 89 c3             	mov    %rax,%rbx
  40c920:	0f 85 06 ff ff ff    	jne    40c82c <__sprintf_chk@plt+0x9f9c>
  40c926:	bf 24 5f 41 00       	mov    $0x415f24,%edi
  40c92b:	e8 90 58 ff ff       	callq  4021c0 <getenv@plt>
  40c930:	48 85 c0             	test   %rax,%rax
  40c933:	48 89 c3             	mov    %rax,%rbx
  40c936:	0f 85 f0 fe ff ff    	jne    40c82c <__sprintf_chk@plt+0x9f9c>
  40c93c:	bf 2e 5f 41 00       	mov    $0x415f2e,%edi
  40c941:	e8 7a 58 ff ff       	callq  4021c0 <getenv@plt>
  40c946:	48 83 f8 01          	cmp    $0x1,%rax
  40c94a:	48 19 d2             	sbb    %rdx,%rdx
  40c94d:	31 ed                	xor    %ebp,%ebp
  40c94f:	81 e2 00 02 00 00    	and    $0x200,%edx
  40c955:	48 81 c2 00 02 00 00 	add    $0x200,%rdx
  40c95c:	49 89 14 24          	mov    %rdx,(%r12)
  40c960:	e9 ff fe ff ff       	jmpq   40c864 <__sprintf_chk@plt+0x9fd4>
  40c965:	0f 1f 00             	nopl   (%rax)
  40c968:	41 c7 45 00 00 00 00 	movl   $0x0,0x0(%r13)
  40c96f:	00 
  40c970:	49 8b 14 24          	mov    (%r12),%rdx
  40c974:	e9 f1 fe ff ff       	jmpq   40c86a <__sprintf_chk@plt+0x9fda>
  40c979:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40c980:	80 7a ff 42          	cmpb   $0x42,-0x1(%rdx)
  40c984:	74 12                	je     40c998 <__sprintf_chk@plt+0xa108>
  40c986:	40 80 cd 80          	or     $0x80,%bpl
  40c98a:	83 cd 20             	or     $0x20,%ebp
  40c98d:	e9 75 ff ff ff       	jmpq   40c907 <__sprintf_chk@plt+0xa077>
  40c992:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40c998:	81 cd 80 01 00 00    	or     $0x180,%ebp
  40c99e:	80 7a fe 69          	cmpb   $0x69,-0x2(%rdx)
  40c9a2:	0f 85 5f ff ff ff    	jne    40c907 <__sprintf_chk@plt+0xa077>
  40c9a8:	eb e0                	jmp    40c98a <__sprintf_chk@plt+0xa0fa>
  40c9aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40c9b0:	41 54                	push   %r12
  40c9b2:	55                   	push   %rbp
  40c9b3:	89 fd                	mov    %edi,%ebp
  40c9b5:	53                   	push   %rbx
  40c9b6:	48 8b 1d 3b e8 20 00 	mov    0x20e83b(%rip),%rbx        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40c9bd:	48 85 db             	test   %rbx,%rbx
  40c9c0:	75 0f                	jne    40c9d1 <__sprintf_chk@plt+0xa141>
  40c9c2:	eb 2c                	jmp    40c9f0 <__sprintf_chk@plt+0xa160>
  40c9c4:	0f 1f 40 00          	nopl   0x0(%rax)
  40c9c8:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40c9cc:	48 85 db             	test   %rbx,%rbx
  40c9cf:	74 1f                	je     40c9f0 <__sprintf_chk@plt+0xa160>
  40c9d1:	39 2b                	cmp    %ebp,(%rbx)
  40c9d3:	75 f3                	jne    40c9c8 <__sprintf_chk@plt+0xa138>
  40c9d5:	31 c0                	xor    %eax,%eax
  40c9d7:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  40c9db:	48 8d 53 10          	lea    0x10(%rbx),%rdx
  40c9df:	5b                   	pop    %rbx
  40c9e0:	5d                   	pop    %rbp
  40c9e1:	41 5c                	pop    %r12
  40c9e3:	48 0f 45 c2          	cmovne %rdx,%rax
  40c9e7:	c3                   	retq   
  40c9e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40c9ef:	00 
  40c9f0:	89 ef                	mov    %ebp,%edi
  40c9f2:	41 bc 19 69 41 00    	mov    $0x416919,%r12d
  40c9f8:	e8 33 59 ff ff       	callq  402330 <getpwuid@plt>
  40c9fd:	48 85 c0             	test   %rax,%rax
  40ca00:	bf 11 00 00 00       	mov    $0x11,%edi
  40ca05:	74 0f                	je     40ca16 <__sprintf_chk@plt+0xa186>
  40ca07:	4c 8b 20             	mov    (%rax),%r12
  40ca0a:	4c 89 e7             	mov    %r12,%rdi
  40ca0d:	e8 6e 59 ff ff       	callq  402380 <strlen@plt>
  40ca12:	48 8d 78 11          	lea    0x11(%rax),%rdi
  40ca16:	e8 25 42 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40ca1b:	48 8d 78 10          	lea    0x10(%rax),%rdi
  40ca1f:	89 28                	mov    %ebp,(%rax)
  40ca21:	4c 89 e6             	mov    %r12,%rsi
  40ca24:	48 89 c3             	mov    %rax,%rbx
  40ca27:	e8 34 58 ff ff       	callq  402260 <strcpy@plt>
  40ca2c:	48 8b 05 c5 e7 20 00 	mov    0x20e7c5(%rip),%rax        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40ca33:	48 89 1d be e7 20 00 	mov    %rbx,0x20e7be(%rip)        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40ca3a:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40ca3e:	eb 95                	jmp    40c9d5 <__sprintf_chk@plt+0xa145>
  40ca40:	41 54                	push   %r12
  40ca42:	49 89 fc             	mov    %rdi,%r12
  40ca45:	55                   	push   %rbp
  40ca46:	53                   	push   %rbx
  40ca47:	48 8b 1d aa e7 20 00 	mov    0x20e7aa(%rip),%rbx        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40ca4e:	48 85 db             	test   %rbx,%rbx
  40ca51:	74 3d                	je     40ca90 <__sprintf_chk@plt+0xa200>
  40ca53:	0f b6 2f             	movzbl (%rdi),%ebp
  40ca56:	eb 11                	jmp    40ca69 <__sprintf_chk@plt+0xa1d9>
  40ca58:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40ca5f:	00 
  40ca60:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40ca64:	48 85 db             	test   %rbx,%rbx
  40ca67:	74 27                	je     40ca90 <__sprintf_chk@plt+0xa200>
  40ca69:	40 38 6b 10          	cmp    %bpl,0x10(%rbx)
  40ca6d:	75 f1                	jne    40ca60 <__sprintf_chk@plt+0xa1d0>
  40ca6f:	48 8d 7b 10          	lea    0x10(%rbx),%rdi
  40ca73:	4c 89 e6             	mov    %r12,%rsi
  40ca76:	e8 d5 5a ff ff       	callq  402550 <strcmp@plt>
  40ca7b:	85 c0                	test   %eax,%eax
  40ca7d:	75 e1                	jne    40ca60 <__sprintf_chk@plt+0xa1d0>
  40ca7f:	48 89 d8             	mov    %rbx,%rax
  40ca82:	5b                   	pop    %rbx
  40ca83:	5d                   	pop    %rbp
  40ca84:	41 5c                	pop    %r12
  40ca86:	c3                   	retq   
  40ca87:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40ca8e:	00 00 
  40ca90:	48 8b 1d 59 e7 20 00 	mov    0x20e759(%rip),%rbx        # 61b1f0 <stderr@@GLIBC_2.2.5+0xba0>
  40ca97:	48 85 db             	test   %rbx,%rbx
  40ca9a:	74 34                	je     40cad0 <__sprintf_chk@plt+0xa240>
  40ca9c:	41 0f b6 2c 24       	movzbl (%r12),%ebp
  40caa1:	eb 0e                	jmp    40cab1 <__sprintf_chk@plt+0xa221>
  40caa3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40caa8:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40caac:	48 85 db             	test   %rbx,%rbx
  40caaf:	74 1f                	je     40cad0 <__sprintf_chk@plt+0xa240>
  40cab1:	40 38 6b 10          	cmp    %bpl,0x10(%rbx)
  40cab5:	75 f1                	jne    40caa8 <__sprintf_chk@plt+0xa218>
  40cab7:	48 8d 7b 10          	lea    0x10(%rbx),%rdi
  40cabb:	4c 89 e6             	mov    %r12,%rsi
  40cabe:	e8 8d 5a ff ff       	callq  402550 <strcmp@plt>
  40cac3:	85 c0                	test   %eax,%eax
  40cac5:	75 e1                	jne    40caa8 <__sprintf_chk@plt+0xa218>
  40cac7:	5b                   	pop    %rbx
  40cac8:	5d                   	pop    %rbp
  40cac9:	31 c0                	xor    %eax,%eax
  40cacb:	41 5c                	pop    %r12
  40cacd:	c3                   	retq   
  40cace:	66 90                	xchg   %ax,%ax
  40cad0:	4c 89 e7             	mov    %r12,%rdi
  40cad3:	e8 a8 5a ff ff       	callq  402580 <getpwnam@plt>
  40cad8:	4c 89 e7             	mov    %r12,%rdi
  40cadb:	48 89 c5             	mov    %rax,%rbp
  40cade:	e8 9d 58 ff ff       	callq  402380 <strlen@plt>
  40cae3:	48 8d 78 11          	lea    0x11(%rax),%rdi
  40cae7:	e8 54 41 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40caec:	48 8d 78 10          	lea    0x10(%rax),%rdi
  40caf0:	4c 89 e6             	mov    %r12,%rsi
  40caf3:	48 89 c3             	mov    %rax,%rbx
  40caf6:	e8 65 57 ff ff       	callq  402260 <strcpy@plt>
  40cafb:	48 85 ed             	test   %rbp,%rbp
  40cafe:	74 1c                	je     40cb1c <__sprintf_chk@plt+0xa28c>
  40cb00:	8b 45 10             	mov    0x10(%rbp),%eax
  40cb03:	89 03                	mov    %eax,(%rbx)
  40cb05:	48 8b 05 ec e6 20 00 	mov    0x20e6ec(%rip),%rax        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40cb0c:	48 89 1d e5 e6 20 00 	mov    %rbx,0x20e6e5(%rip)        # 61b1f8 <stderr@@GLIBC_2.2.5+0xba8>
  40cb13:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40cb17:	e9 63 ff ff ff       	jmpq   40ca7f <__sprintf_chk@plt+0xa1ef>
  40cb1c:	48 8b 05 cd e6 20 00 	mov    0x20e6cd(%rip),%rax        # 61b1f0 <stderr@@GLIBC_2.2.5+0xba0>
  40cb23:	48 89 1d c6 e6 20 00 	mov    %rbx,0x20e6c6(%rip)        # 61b1f0 <stderr@@GLIBC_2.2.5+0xba0>
  40cb2a:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40cb2e:	31 c0                	xor    %eax,%eax
  40cb30:	e9 4d ff ff ff       	jmpq   40ca82 <__sprintf_chk@plt+0xa1f2>
  40cb35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40cb3c:	00 00 00 00 
  40cb40:	41 54                	push   %r12
  40cb42:	55                   	push   %rbp
  40cb43:	89 fd                	mov    %edi,%ebp
  40cb45:	53                   	push   %rbx
  40cb46:	48 8b 1d 9b e6 20 00 	mov    0x20e69b(%rip),%rbx        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cb4d:	48 85 db             	test   %rbx,%rbx
  40cb50:	75 0f                	jne    40cb61 <__sprintf_chk@plt+0xa2d1>
  40cb52:	eb 2c                	jmp    40cb80 <__sprintf_chk@plt+0xa2f0>
  40cb54:	0f 1f 40 00          	nopl   0x0(%rax)
  40cb58:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40cb5c:	48 85 db             	test   %rbx,%rbx
  40cb5f:	74 1f                	je     40cb80 <__sprintf_chk@plt+0xa2f0>
  40cb61:	39 2b                	cmp    %ebp,(%rbx)
  40cb63:	75 f3                	jne    40cb58 <__sprintf_chk@plt+0xa2c8>
  40cb65:	31 c0                	xor    %eax,%eax
  40cb67:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  40cb6b:	48 8d 53 10          	lea    0x10(%rbx),%rdx
  40cb6f:	5b                   	pop    %rbx
  40cb70:	5d                   	pop    %rbp
  40cb71:	41 5c                	pop    %r12
  40cb73:	48 0f 45 c2          	cmovne %rdx,%rax
  40cb77:	c3                   	retq   
  40cb78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40cb7f:	00 
  40cb80:	89 ef                	mov    %ebp,%edi
  40cb82:	41 bc 19 69 41 00    	mov    $0x416919,%r12d
  40cb88:	e8 53 58 ff ff       	callq  4023e0 <getgrgid@plt>
  40cb8d:	48 85 c0             	test   %rax,%rax
  40cb90:	bf 11 00 00 00       	mov    $0x11,%edi
  40cb95:	74 0f                	je     40cba6 <__sprintf_chk@plt+0xa316>
  40cb97:	4c 8b 20             	mov    (%rax),%r12
  40cb9a:	4c 89 e7             	mov    %r12,%rdi
  40cb9d:	e8 de 57 ff ff       	callq  402380 <strlen@plt>
  40cba2:	48 8d 78 11          	lea    0x11(%rax),%rdi
  40cba6:	e8 95 40 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40cbab:	48 8d 78 10          	lea    0x10(%rax),%rdi
  40cbaf:	89 28                	mov    %ebp,(%rax)
  40cbb1:	4c 89 e6             	mov    %r12,%rsi
  40cbb4:	48 89 c3             	mov    %rax,%rbx
  40cbb7:	e8 a4 56 ff ff       	callq  402260 <strcpy@plt>
  40cbbc:	48 8b 05 25 e6 20 00 	mov    0x20e625(%rip),%rax        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cbc3:	48 89 1d 1e e6 20 00 	mov    %rbx,0x20e61e(%rip)        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cbca:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40cbce:	eb 95                	jmp    40cb65 <__sprintf_chk@plt+0xa2d5>
  40cbd0:	41 54                	push   %r12
  40cbd2:	49 89 fc             	mov    %rdi,%r12
  40cbd5:	55                   	push   %rbp
  40cbd6:	53                   	push   %rbx
  40cbd7:	48 8b 1d 0a e6 20 00 	mov    0x20e60a(%rip),%rbx        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cbde:	48 85 db             	test   %rbx,%rbx
  40cbe1:	74 3d                	je     40cc20 <__sprintf_chk@plt+0xa390>
  40cbe3:	0f b6 2f             	movzbl (%rdi),%ebp
  40cbe6:	eb 11                	jmp    40cbf9 <__sprintf_chk@plt+0xa369>
  40cbe8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40cbef:	00 
  40cbf0:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40cbf4:	48 85 db             	test   %rbx,%rbx
  40cbf7:	74 27                	je     40cc20 <__sprintf_chk@plt+0xa390>
  40cbf9:	40 38 6b 10          	cmp    %bpl,0x10(%rbx)
  40cbfd:	75 f1                	jne    40cbf0 <__sprintf_chk@plt+0xa360>
  40cbff:	48 8d 7b 10          	lea    0x10(%rbx),%rdi
  40cc03:	4c 89 e6             	mov    %r12,%rsi
  40cc06:	e8 45 59 ff ff       	callq  402550 <strcmp@plt>
  40cc0b:	85 c0                	test   %eax,%eax
  40cc0d:	75 e1                	jne    40cbf0 <__sprintf_chk@plt+0xa360>
  40cc0f:	48 89 d8             	mov    %rbx,%rax
  40cc12:	5b                   	pop    %rbx
  40cc13:	5d                   	pop    %rbp
  40cc14:	41 5c                	pop    %r12
  40cc16:	c3                   	retq   
  40cc17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40cc1e:	00 00 
  40cc20:	48 8b 1d b9 e5 20 00 	mov    0x20e5b9(%rip),%rbx        # 61b1e0 <stderr@@GLIBC_2.2.5+0xb90>
  40cc27:	48 85 db             	test   %rbx,%rbx
  40cc2a:	74 34                	je     40cc60 <__sprintf_chk@plt+0xa3d0>
  40cc2c:	41 0f b6 2c 24       	movzbl (%r12),%ebp
  40cc31:	eb 0e                	jmp    40cc41 <__sprintf_chk@plt+0xa3b1>
  40cc33:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40cc38:	48 8b 5b 08          	mov    0x8(%rbx),%rbx
  40cc3c:	48 85 db             	test   %rbx,%rbx
  40cc3f:	74 1f                	je     40cc60 <__sprintf_chk@plt+0xa3d0>
  40cc41:	40 38 6b 10          	cmp    %bpl,0x10(%rbx)
  40cc45:	75 f1                	jne    40cc38 <__sprintf_chk@plt+0xa3a8>
  40cc47:	48 8d 7b 10          	lea    0x10(%rbx),%rdi
  40cc4b:	4c 89 e6             	mov    %r12,%rsi
  40cc4e:	e8 fd 58 ff ff       	callq  402550 <strcmp@plt>
  40cc53:	85 c0                	test   %eax,%eax
  40cc55:	75 e1                	jne    40cc38 <__sprintf_chk@plt+0xa3a8>
  40cc57:	5b                   	pop    %rbx
  40cc58:	5d                   	pop    %rbp
  40cc59:	31 c0                	xor    %eax,%eax
  40cc5b:	41 5c                	pop    %r12
  40cc5d:	c3                   	retq   
  40cc5e:	66 90                	xchg   %ax,%ax
  40cc60:	4c 89 e7             	mov    %r12,%rdi
  40cc63:	e8 68 59 ff ff       	callq  4025d0 <getgrnam@plt>
  40cc68:	4c 89 e7             	mov    %r12,%rdi
  40cc6b:	48 89 c5             	mov    %rax,%rbp
  40cc6e:	e8 0d 57 ff ff       	callq  402380 <strlen@plt>
  40cc73:	48 8d 78 11          	lea    0x11(%rax),%rdi
  40cc77:	e8 c4 3f 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40cc7c:	48 8d 78 10          	lea    0x10(%rax),%rdi
  40cc80:	4c 89 e6             	mov    %r12,%rsi
  40cc83:	48 89 c3             	mov    %rax,%rbx
  40cc86:	e8 d5 55 ff ff       	callq  402260 <strcpy@plt>
  40cc8b:	48 85 ed             	test   %rbp,%rbp
  40cc8e:	74 1c                	je     40ccac <__sprintf_chk@plt+0xa41c>
  40cc90:	8b 45 10             	mov    0x10(%rbp),%eax
  40cc93:	89 03                	mov    %eax,(%rbx)
  40cc95:	48 8b 05 4c e5 20 00 	mov    0x20e54c(%rip),%rax        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cc9c:	48 89 1d 45 e5 20 00 	mov    %rbx,0x20e545(%rip)        # 61b1e8 <stderr@@GLIBC_2.2.5+0xb98>
  40cca3:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40cca7:	e9 63 ff ff ff       	jmpq   40cc0f <__sprintf_chk@plt+0xa37f>
  40ccac:	48 8b 05 2d e5 20 00 	mov    0x20e52d(%rip),%rax        # 61b1e0 <stderr@@GLIBC_2.2.5+0xb90>
  40ccb3:	48 89 1d 26 e5 20 00 	mov    %rbx,0x20e526(%rip)        # 61b1e0 <stderr@@GLIBC_2.2.5+0xb90>
  40ccba:	48 89 43 08          	mov    %rax,0x8(%rbx)
  40ccbe:	31 c0                	xor    %eax,%eax
  40ccc0:	e9 4d ff ff ff       	jmpq   40cc12 <__sprintf_chk@plt+0xa382>
  40ccc5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40cccc:	00 00 00 
  40cccf:	90                   	nop
  40ccd0:	48 85 ff             	test   %rdi,%rdi
  40ccd3:	48 8d 4e 14          	lea    0x14(%rsi),%rcx
  40ccd7:	c6 46 14 00          	movb   $0x0,0x14(%rsi)
  40ccdb:	48 be 67 66 66 66 66 	movabs $0x6666666666666667,%rsi
  40cce2:	66 66 66 
  40cce5:	78 41                	js     40cd28 <__sprintf_chk@plt+0xa498>
  40cce7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40ccee:	00 00 
  40ccf0:	48 89 f8             	mov    %rdi,%rax
  40ccf3:	48 83 e9 01          	sub    $0x1,%rcx
  40ccf7:	48 f7 ee             	imul   %rsi
  40ccfa:	48 89 f8             	mov    %rdi,%rax
  40ccfd:	48 c1 f8 3f          	sar    $0x3f,%rax
  40cd01:	48 c1 fa 02          	sar    $0x2,%rdx
  40cd05:	48 29 c2             	sub    %rax,%rdx
  40cd08:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  40cd0c:	48 01 c0             	add    %rax,%rax
  40cd0f:	48 29 c7             	sub    %rax,%rdi
  40cd12:	83 c7 30             	add    $0x30,%edi
  40cd15:	48 85 d2             	test   %rdx,%rdx
  40cd18:	40 88 39             	mov    %dil,(%rcx)
  40cd1b:	48 89 d7             	mov    %rdx,%rdi
  40cd1e:	75 d0                	jne    40ccf0 <__sprintf_chk@plt+0xa460>
  40cd20:	48 89 c8             	mov    %rcx,%rax
  40cd23:	c3                   	retq   
  40cd24:	0f 1f 40 00          	nopl   0x0(%rax)
  40cd28:	49 89 f0             	mov    %rsi,%r8
  40cd2b:	be 30 00 00 00       	mov    $0x30,%esi
  40cd30:	48 89 f8             	mov    %rdi,%rax
  40cd33:	48 83 e9 01          	sub    $0x1,%rcx
  40cd37:	49 f7 e8             	imul   %r8
  40cd3a:	48 89 f8             	mov    %rdi,%rax
  40cd3d:	48 c1 f8 3f          	sar    $0x3f,%rax
  40cd41:	48 c1 fa 02          	sar    $0x2,%rdx
  40cd45:	48 29 c2             	sub    %rax,%rdx
  40cd48:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  40cd4c:	8d 04 46             	lea    (%rsi,%rax,2),%eax
  40cd4f:	29 f8                	sub    %edi,%eax
  40cd51:	48 85 d2             	test   %rdx,%rdx
  40cd54:	48 89 d7             	mov    %rdx,%rdi
  40cd57:	88 01                	mov    %al,(%rcx)
  40cd59:	75 d5                	jne    40cd30 <__sprintf_chk@plt+0xa4a0>
  40cd5b:	48 89 c8             	mov    %rcx,%rax
  40cd5e:	48 83 e9 01          	sub    $0x1,%rcx
  40cd62:	c6 40 ff 2d          	movb   $0x2d,-0x1(%rax)
  40cd66:	48 89 c8             	mov    %rcx,%rax
  40cd69:	c3                   	retq   
  40cd6a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40cd70:	48 8d 4e 14          	lea    0x14(%rsi),%rcx
  40cd74:	c6 46 14 00          	movb   $0x0,0x14(%rsi)
  40cd78:	48 be cd cc cc cc cc 	movabs $0xcccccccccccccccd,%rsi
  40cd7f:	cc cc cc 
  40cd82:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40cd88:	48 89 f8             	mov    %rdi,%rax
  40cd8b:	48 83 e9 01          	sub    $0x1,%rcx
  40cd8f:	48 f7 e6             	mul    %rsi
  40cd92:	48 c1 ea 03          	shr    $0x3,%rdx
  40cd96:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  40cd9a:	48 01 c0             	add    %rax,%rax
  40cd9d:	48 29 c7             	sub    %rax,%rdi
  40cda0:	83 c7 30             	add    $0x30,%edi
  40cda3:	48 85 d2             	test   %rdx,%rdx
  40cda6:	40 88 39             	mov    %dil,(%rcx)
  40cda9:	48 89 d7             	mov    %rdx,%rdi
  40cdac:	75 da                	jne    40cd88 <__sprintf_chk@plt+0xa4f8>
  40cdae:	48 89 c8             	mov    %rcx,%rax
  40cdb1:	c3                   	retq   
  40cdb2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40cdb9:	00 00 00 
  40cdbc:	0f 1f 40 00          	nopl   0x0(%rax)
  40cdc0:	41 57                	push   %r15
  40cdc2:	41 56                	push   %r14
  40cdc4:	41 55                	push   %r13
  40cdc6:	49 89 f5             	mov    %rsi,%r13
  40cdc9:	41 54                	push   %r12
  40cdcb:	49 89 cc             	mov    %rcx,%r12
  40cdce:	55                   	push   %rbp
  40cdcf:	53                   	push   %rbx
  40cdd0:	44 89 cb             	mov    %r9d,%ebx
  40cdd3:	48 83 ec 38          	sub    $0x38,%rsp
  40cdd7:	48 89 7c 24 08       	mov    %rdi,0x8(%rsp)
  40cddc:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  40cde1:	44 89 44 24 28       	mov    %r8d,0x28(%rsp)
  40cde6:	e8 95 55 ff ff       	callq  402380 <strlen@plt>
  40cdeb:	f6 c3 02             	test   $0x2,%bl
  40cdee:	49 89 c6             	mov    %rax,%r14
  40cdf1:	48 89 c5             	mov    %rax,%rbp
  40cdf4:	0f 84 66 01 00 00    	je     40cf60 <__sprintf_chk@plt+0xa6d0>
  40cdfa:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40ce01:	00 00 
  40ce03:	49 89 ef             	mov    %rbp,%r15
  40ce06:	31 ed                	xor    %ebp,%ebp
  40ce08:	49 8b 04 24          	mov    (%r12),%rax
  40ce0c:	49 39 c7             	cmp    %rax,%r15
  40ce0f:	0f 86 15 01 00 00    	jbe    40cf2a <__sprintf_chk@plt+0xa69a>
  40ce15:	49 89 c6             	mov    %rax,%r14
  40ce18:	31 c9                	xor    %ecx,%ecx
  40ce1a:	49 89 04 24          	mov    %rax,(%r12)
  40ce1e:	8b 44 24 28          	mov    0x28(%rsp),%eax
  40ce22:	85 c0                	test   %eax,%eax
  40ce24:	0f 84 22 01 00 00    	je     40cf4c <__sprintf_chk@plt+0xa6bc>
  40ce2a:	45 31 e4             	xor    %r12d,%r12d
  40ce2d:	83 f8 01             	cmp    $0x1,%eax
  40ce30:	74 0c                	je     40ce3e <__sprintf_chk@plt+0xa5ae>
  40ce32:	49 89 cc             	mov    %rcx,%r12
  40ce35:	83 e1 01             	and    $0x1,%ecx
  40ce38:	49 d1 ec             	shr    %r12
  40ce3b:	4c 01 e1             	add    %r12,%rcx
  40ce3e:	31 c0                	xor    %eax,%eax
  40ce40:	f6 c3 04             	test   $0x4,%bl
  40ce43:	48 0f 45 c8          	cmovne %rax,%rcx
  40ce47:	83 e3 08             	and    $0x8,%ebx
  40ce4a:	4c 0f 45 e0          	cmovne %rax,%r12
  40ce4e:	48 83 7c 24 18 00    	cmpq   $0x0,0x18(%rsp)
  40ce54:	0f 84 9f 00 00 00    	je     40cef9 <__sprintf_chk@plt+0xa669>
  40ce5a:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40ce5f:	48 85 c9             	test   %rcx,%rcx
  40ce62:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
  40ce66:	49 8d 5c 05 ff       	lea    -0x1(%r13,%rax,1),%rbx
  40ce6b:	74 2a                	je     40ce97 <__sprintf_chk@plt+0xa607>
  40ce6d:	49 39 dd             	cmp    %rbx,%r13
  40ce70:	73 25                	jae    40ce97 <__sprintf_chk@plt+0xa607>
  40ce72:	31 c0                	xor    %eax,%eax
  40ce74:	eb 13                	jmp    40ce89 <__sprintf_chk@plt+0xa5f9>
  40ce76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40ce7d:	00 00 00 
  40ce80:	48 83 c0 01          	add    $0x1,%rax
  40ce84:	49 39 dd             	cmp    %rbx,%r13
  40ce87:	74 0e                	je     40ce97 <__sprintf_chk@plt+0xa607>
  40ce89:	49 83 c5 01          	add    $0x1,%r13
  40ce8d:	48 39 c2             	cmp    %rax,%rdx
  40ce90:	41 c6 45 ff 20       	movb   $0x20,-0x1(%r13)
  40ce95:	75 e9                	jne    40ce80 <__sprintf_chk@plt+0xa5f0>
  40ce97:	48 89 da             	mov    %rbx,%rdx
  40ce9a:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40ce9f:	41 c6 45 00 00       	movb   $0x0,0x0(%r13)
  40cea4:	4c 29 ea             	sub    %r13,%rdx
  40cea7:	4c 89 ef             	mov    %r13,%rdi
  40ceaa:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
  40ceaf:	4c 39 f2             	cmp    %r14,%rdx
  40ceb2:	49 0f 47 d6          	cmova  %r14,%rdx
  40ceb6:	e8 95 58 ff ff       	callq  402750 <mempcpy@plt>
  40cebb:	4d 85 e4             	test   %r12,%r12
  40cebe:	48 89 c2             	mov    %rax,%rdx
  40cec1:	49 8d 74 24 ff       	lea    -0x1(%r12),%rsi
  40cec6:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40cecb:	74 29                	je     40cef6 <__sprintf_chk@plt+0xa666>
  40cecd:	48 39 c3             	cmp    %rax,%rbx
  40ced0:	76 24                	jbe    40cef6 <__sprintf_chk@plt+0xa666>
  40ced2:	31 c0                	xor    %eax,%eax
  40ced4:	eb 13                	jmp    40cee9 <__sprintf_chk@plt+0xa659>
  40ced6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40cedd:	00 00 00 
  40cee0:	48 83 c0 01          	add    $0x1,%rax
  40cee4:	48 39 da             	cmp    %rbx,%rdx
  40cee7:	74 0d                	je     40cef6 <__sprintf_chk@plt+0xa666>
  40cee9:	48 83 c2 01          	add    $0x1,%rdx
  40ceed:	48 39 c6             	cmp    %rax,%rsi
  40cef0:	c6 42 ff 20          	movb   $0x20,-0x1(%rdx)
  40cef4:	75 ea                	jne    40cee0 <__sprintf_chk@plt+0xa650>
  40cef6:	c6 02 00             	movb   $0x0,(%rdx)
  40cef9:	4c 01 f1             	add    %r14,%rcx
  40cefc:	49 01 cc             	add    %rcx,%r12
  40ceff:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40cf04:	e8 e7 52 ff ff       	callq  4021f0 <free@plt>
  40cf09:	48 89 ef             	mov    %rbp,%rdi
  40cf0c:	e8 df 52 ff ff       	callq  4021f0 <free@plt>
  40cf11:	48 83 c4 38          	add    $0x38,%rsp
  40cf15:	4c 89 e0             	mov    %r12,%rax
  40cf18:	5b                   	pop    %rbx
  40cf19:	5d                   	pop    %rbp
  40cf1a:	41 5c                	pop    %r12
  40cf1c:	41 5d                	pop    %r13
  40cf1e:	41 5e                	pop    %r14
  40cf20:	41 5f                	pop    %r15
  40cf22:	c3                   	retq   
  40cf23:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40cf28:	31 ed                	xor    %ebp,%ebp
  40cf2a:	49 39 c7             	cmp    %rax,%r15
  40cf2d:	0f 83 4d 02 00 00    	jae    40d180 <__sprintf_chk@plt+0xa8f0>
  40cf33:	4c 29 f8             	sub    %r15,%rax
  40cf36:	48 89 c1             	mov    %rax,%rcx
  40cf39:	4c 89 f8             	mov    %r15,%rax
  40cf3c:	49 89 04 24          	mov    %rax,(%r12)
  40cf40:	8b 44 24 28          	mov    0x28(%rsp),%eax
  40cf44:	85 c0                	test   %eax,%eax
  40cf46:	0f 85 de fe ff ff    	jne    40ce2a <__sprintf_chk@plt+0xa59a>
  40cf4c:	49 89 cc             	mov    %rcx,%r12
  40cf4f:	31 c9                	xor    %ecx,%ecx
  40cf51:	e9 e8 fe ff ff       	jmpq   40ce3e <__sprintf_chk@plt+0xa5ae>
  40cf56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40cf5d:	00 00 00 
  40cf60:	e8 0b 54 ff ff       	callq  402370 <__ctype_get_mb_cur_max@plt>
  40cf65:	48 83 f8 01          	cmp    $0x1,%rax
  40cf69:	0f 86 8b fe ff ff    	jbe    40cdfa <__sprintf_chk@plt+0xa56a>
  40cf6f:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40cf74:	31 d2                	xor    %edx,%edx
  40cf76:	31 ff                	xor    %edi,%edi
  40cf78:	e8 53 53 ff ff       	callq  4022d0 <mbstowcs@plt>
  40cf7d:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  40cf81:	75 25                	jne    40cfa8 <__sprintf_chk@plt+0xa718>
  40cf83:	f6 c3 01             	test   $0x1,%bl
  40cf86:	0f 85 c1 01 00 00    	jne    40d14d <__sprintf_chk@plt+0xa8bd>
  40cf8c:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40cf93:	00 00 
  40cf95:	31 ed                	xor    %ebp,%ebp
  40cf97:	49 c7 c4 ff ff ff ff 	mov    $0xffffffffffffffff,%r12
  40cf9e:	e9 5c ff ff ff       	jmpq   40ceff <__sprintf_chk@plt+0xa66f>
  40cfa3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40cfa8:	48 83 c0 01          	add    $0x1,%rax
  40cfac:	4c 8d 3c 85 00 00 00 	lea    0x0(,%rax,4),%r15
  40cfb3:	00 
  40cfb4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40cfb9:	4c 89 ff             	mov    %r15,%rdi
  40cfbc:	e8 7f 56 ff ff       	callq  402640 <malloc@plt>
  40cfc1:	48 85 c0             	test   %rax,%rax
  40cfc4:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40cfc9:	0f 84 91 01 00 00    	je     40d160 <__sprintf_chk@plt+0xa8d0>
  40cfcf:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  40cfd4:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40cfd9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40cfde:	e8 ed 52 ff ff       	callq  4022d0 <mbstowcs@plt>
  40cfe3:	48 85 c0             	test   %rax,%rax
  40cfe6:	0f 84 17 fe ff ff    	je     40ce03 <__sprintf_chk@plt+0xa573>
  40cfec:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40cff1:	42 c7 44 38 fc 00 00 	movl   $0x0,-0x4(%rax,%r15,1)
  40cff8:	00 00 
  40cffa:	8b 38                	mov    (%rax),%edi
  40cffc:	85 ff                	test   %edi,%edi
  40cffe:	0f 84 02 01 00 00    	je     40d106 <__sprintf_chk@plt+0xa876>
  40d004:	49 89 c7             	mov    %rax,%r15
  40d007:	c6 44 24 2f 00       	movb   $0x0,0x2f(%rsp)
  40d00c:	0f 1f 40 00          	nopl   0x0(%rax)
  40d010:	e8 2b 58 ff ff       	callq  402840 <iswprint@plt>
  40d015:	85 c0                	test   %eax,%eax
  40d017:	75 0c                	jne    40d025 <__sprintf_chk@plt+0xa795>
  40d019:	41 c7 07 fd ff 00 00 	movl   $0xfffd,(%r15)
  40d020:	c6 44 24 2f 01       	movb   $0x1,0x2f(%rsp)
  40d025:	49 83 c7 04          	add    $0x4,%r15
  40d029:	41 8b 3f             	mov    (%r15),%edi
  40d02c:	85 ff                	test   %edi,%edi
  40d02e:	75 e0                	jne    40d010 <__sprintf_chk@plt+0xa780>
  40d030:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  40d035:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40d03a:	e8 71 52 ff ff       	callq  4022b0 <wcswidth@plt>
  40d03f:	80 7c 24 2f 00       	cmpb   $0x0,0x2f(%rsp)
  40d044:	4c 63 f8             	movslq %eax,%r15
  40d047:	0f 84 cb 00 00 00    	je     40d118 <__sprintf_chk@plt+0xa888>
  40d04d:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  40d052:	31 d2                	xor    %edx,%edx
  40d054:	31 ff                	xor    %edi,%edi
  40d056:	e8 65 57 ff ff       	callq  4027c0 <wcstombs@plt>
  40d05b:	48 83 c0 01          	add    $0x1,%rax
  40d05f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40d064:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  40d069:	e8 d2 55 ff ff       	callq  402640 <malloc@plt>
  40d06e:	48 85 c0             	test   %rax,%rax
  40d071:	48 89 c5             	mov    %rax,%rbp
  40d074:	0f 84 be 00 00 00    	je     40d138 <__sprintf_chk@plt+0xa8a8>
  40d07a:	49 8b 04 24          	mov    (%r12),%rax
  40d07e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40d083:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d088:	8b 38                	mov    (%rax),%edi
  40d08a:	85 ff                	test   %edi,%edi
  40d08c:	0f 84 e1 00 00 00    	je     40d173 <__sprintf_chk@plt+0xa8e3>
  40d092:	49 89 c6             	mov    %rax,%r14
  40d095:	45 31 ff             	xor    %r15d,%r15d
  40d098:	eb 20                	jmp    40d0ba <__sprintf_chk@plt+0xa82a>
  40d09a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40d0a0:	48 98                	cltq   
  40d0a2:	4c 01 f8             	add    %r15,%rax
  40d0a5:	48 39 44 24 08       	cmp    %rax,0x8(%rsp)
  40d0aa:	72 34                	jb     40d0e0 <__sprintf_chk@plt+0xa850>
  40d0ac:	49 83 c6 04          	add    $0x4,%r14
  40d0b0:	41 8b 3e             	mov    (%r14),%edi
  40d0b3:	49 89 c7             	mov    %rax,%r15
  40d0b6:	85 ff                	test   %edi,%edi
  40d0b8:	74 26                	je     40d0e0 <__sprintf_chk@plt+0xa850>
  40d0ba:	e8 71 55 ff ff       	callq  402630 <wcwidth@plt>
  40d0bf:	83 f8 ff             	cmp    $0xffffffff,%eax
  40d0c2:	75 dc                	jne    40d0a0 <__sprintf_chk@plt+0xa810>
  40d0c4:	b8 01 00 00 00       	mov    $0x1,%eax
  40d0c9:	41 c7 06 fd ff 00 00 	movl   $0xfffd,(%r14)
  40d0d0:	4c 01 f8             	add    %r15,%rax
  40d0d3:	48 39 44 24 08       	cmp    %rax,0x8(%rsp)
  40d0d8:	73 d2                	jae    40d0ac <__sprintf_chk@plt+0xa81c>
  40d0da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40d0e0:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  40d0e5:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  40d0ea:	48 89 ef             	mov    %rbp,%rdi
  40d0ed:	41 c7 06 00 00 00 00 	movl   $0x0,(%r14)
  40d0f4:	e8 c7 56 ff ff       	callq  4027c0 <wcstombs@plt>
  40d0f9:	48 89 6c 24 08       	mov    %rbp,0x8(%rsp)
  40d0fe:	49 89 c6             	mov    %rax,%r14
  40d101:	e9 02 fd ff ff       	jmpq   40ce08 <__sprintf_chk@plt+0xa578>
  40d106:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  40d10b:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  40d110:	e8 9b 51 ff ff       	callq  4022b0 <wcswidth@plt>
  40d115:	4c 63 f8             	movslq %eax,%r15
  40d118:	49 8b 04 24          	mov    (%r12),%rax
  40d11c:	49 39 c7             	cmp    %rax,%r15
  40d11f:	0f 86 03 fe ff ff    	jbe    40cf28 <__sprintf_chk@plt+0xa698>
  40d125:	48 8d 45 01          	lea    0x1(%rbp),%rax
  40d129:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40d12e:	e9 31 ff ff ff       	jmpq   40d064 <__sprintf_chk@plt+0xa7d4>
  40d133:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40d138:	f6 c3 01             	test   $0x1,%bl
  40d13b:	0f 85 c7 fc ff ff    	jne    40ce08 <__sprintf_chk@plt+0xa578>
  40d141:	49 c7 c4 ff ff ff ff 	mov    $0xffffffffffffffff,%r12
  40d148:	e9 b2 fd ff ff       	jmpq   40ceff <__sprintf_chk@plt+0xa66f>
  40d14d:	4d 89 f7             	mov    %r14,%r15
  40d150:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40d157:	00 00 
  40d159:	31 ed                	xor    %ebp,%ebp
  40d15b:	e9 a8 fc ff ff       	jmpq   40ce08 <__sprintf_chk@plt+0xa578>
  40d160:	f6 c3 01             	test   $0x1,%bl
  40d163:	0f 84 23 fe ff ff    	je     40cf8c <__sprintf_chk@plt+0xa6fc>
  40d169:	4d 89 f7             	mov    %r14,%r15
  40d16c:	31 ed                	xor    %ebp,%ebp
  40d16e:	e9 95 fc ff ff       	jmpq   40ce08 <__sprintf_chk@plt+0xa578>
  40d173:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
  40d178:	45 31 ff             	xor    %r15d,%r15d
  40d17b:	e9 60 ff ff ff       	jmpq   40d0e0 <__sprintf_chk@plt+0xa850>
  40d180:	4c 89 f8             	mov    %r15,%rax
  40d183:	31 c9                	xor    %ecx,%ecx
  40d185:	e9 90 fc ff ff       	jmpq   40ce1a <__sprintf_chk@plt+0xa58a>
  40d18a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40d190:	41 57                	push   %r15
  40d192:	49 89 ff             	mov    %rdi,%r15
  40d195:	41 56                	push   %r14
  40d197:	41 55                	push   %r13
  40d199:	41 54                	push   %r12
  40d19b:	45 31 e4             	xor    %r12d,%r12d
  40d19e:	55                   	push   %rbp
  40d19f:	48 89 f5             	mov    %rsi,%rbp
  40d1a2:	53                   	push   %rbx
  40d1a3:	48 83 ec 18          	sub    $0x18,%rsp
  40d1a7:	4c 8b 2e             	mov    (%rsi),%r13
  40d1aa:	89 54 24 08          	mov    %edx,0x8(%rsp)
  40d1ae:	89 4c 24 0c          	mov    %ecx,0xc(%rsp)
  40d1b2:	4c 89 e8             	mov    %r13,%rax
  40d1b5:	eb 0c                	jmp    40d1c3 <__sprintf_chk@plt+0xa933>
  40d1b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40d1be:	00 00 
  40d1c0:	4d 89 f4             	mov    %r14,%r12
  40d1c3:	48 8d 58 01          	lea    0x1(%rax),%rbx
  40d1c7:	4c 89 e7             	mov    %r12,%rdi
  40d1ca:	48 89 de             	mov    %rbx,%rsi
  40d1cd:	e8 0e 55 ff ff       	callq  4026e0 <realloc@plt>
  40d1d2:	48 85 c0             	test   %rax,%rax
  40d1d5:	49 89 c6             	mov    %rax,%r14
  40d1d8:	74 46                	je     40d220 <__sprintf_chk@plt+0xa990>
  40d1da:	44 8b 4c 24 0c       	mov    0xc(%rsp),%r9d
  40d1df:	44 8b 44 24 08       	mov    0x8(%rsp),%r8d
  40d1e4:	48 89 e9             	mov    %rbp,%rcx
  40d1e7:	4c 89 6d 00          	mov    %r13,0x0(%rbp)
  40d1eb:	48 89 da             	mov    %rbx,%rdx
  40d1ee:	48 89 c6             	mov    %rax,%rsi
  40d1f1:	4c 89 ff             	mov    %r15,%rdi
  40d1f4:	e8 c7 fb ff ff       	callq  40cdc0 <__sprintf_chk@plt+0xa530>
  40d1f9:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  40d1fd:	74 31                	je     40d230 <__sprintf_chk@plt+0xa9a0>
  40d1ff:	48 39 c3             	cmp    %rax,%rbx
  40d202:	76 bc                	jbe    40d1c0 <__sprintf_chk@plt+0xa930>
  40d204:	48 83 c4 18          	add    $0x18,%rsp
  40d208:	4c 89 f0             	mov    %r14,%rax
  40d20b:	5b                   	pop    %rbx
  40d20c:	5d                   	pop    %rbp
  40d20d:	41 5c                	pop    %r12
  40d20f:	41 5d                	pop    %r13
  40d211:	41 5e                	pop    %r14
  40d213:	41 5f                	pop    %r15
  40d215:	c3                   	retq   
  40d216:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40d21d:	00 00 00 
  40d220:	4c 89 e7             	mov    %r12,%rdi
  40d223:	e8 c8 4f ff ff       	callq  4021f0 <free@plt>
  40d228:	eb da                	jmp    40d204 <__sprintf_chk@plt+0xa974>
  40d22a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40d230:	4c 89 f7             	mov    %r14,%rdi
  40d233:	45 31 f6             	xor    %r14d,%r14d
  40d236:	e8 b5 4f ff ff       	callq  4021f0 <free@plt>
  40d23b:	eb c7                	jmp    40d204 <__sprintf_chk@plt+0xa974>
  40d23d:	0f 1f 00             	nopl   (%rax)
  40d240:	41 57                	push   %r15
  40d242:	41 89 d7             	mov    %edx,%r15d
  40d245:	41 56                	push   %r14
  40d247:	41 55                	push   %r13
  40d249:	4c 8d 2c 37          	lea    (%rdi,%rsi,1),%r13
  40d24d:	41 54                	push   %r12
  40d24f:	55                   	push   %rbp
  40d250:	48 89 fd             	mov    %rdi,%rbp
  40d253:	53                   	push   %rbx
  40d254:	48 83 ec 28          	sub    $0x28,%rsp
  40d258:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40d25f:	00 00 
  40d261:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40d266:	31 c0                	xor    %eax,%eax
  40d268:	e8 03 51 ff ff       	callq  402370 <__ctype_get_mb_cur_max@plt>
  40d26d:	48 83 f8 01          	cmp    $0x1,%rax
  40d271:	0f 86 f9 00 00 00    	jbe    40d370 <__sprintf_chk@plt+0xaae0>
  40d277:	4c 39 ed             	cmp    %r13,%rbp
  40d27a:	0f 83 86 01 00 00    	jae    40d406 <__sprintf_chk@plt+0xab76>
  40d280:	45 89 fe             	mov    %r15d,%r14d
  40d283:	45 31 e4             	xor    %r12d,%r12d
  40d286:	41 83 e7 01          	and    $0x1,%r15d
  40d28a:	41 83 e6 02          	and    $0x2,%r14d
  40d28e:	eb 1c                	jmp    40d2ac <__sprintf_chk@plt+0xaa1c>
  40d290:	3c 25                	cmp    $0x25,%al
  40d292:	7d 07                	jge    40d29b <__sprintf_chk@plt+0xaa0b>
  40d294:	83 e8 20             	sub    $0x20,%eax
  40d297:	3c 03                	cmp    $0x3,%al
  40d299:	77 28                	ja     40d2c3 <__sprintf_chk@plt+0xaa33>
  40d29b:	48 83 c5 01          	add    $0x1,%rbp
  40d29f:	41 83 c4 01          	add    $0x1,%r12d
  40d2a3:	49 39 ed             	cmp    %rbp,%r13
  40d2a6:	0f 86 55 01 00 00    	jbe    40d401 <__sprintf_chk@plt+0xab71>
  40d2ac:	0f b6 45 00          	movzbl 0x0(%rbp),%eax
  40d2b0:	3c 3f                	cmp    $0x3f,%al
  40d2b2:	7e dc                	jle    40d290 <__sprintf_chk@plt+0xaa00>
  40d2b4:	3c 41                	cmp    $0x41,%al
  40d2b6:	7c 0b                	jl     40d2c3 <__sprintf_chk@plt+0xaa33>
  40d2b8:	3c 5f                	cmp    $0x5f,%al
  40d2ba:	7e df                	jle    40d29b <__sprintf_chk@plt+0xaa0b>
  40d2bc:	83 e8 61             	sub    $0x61,%eax
  40d2bf:	3c 1d                	cmp    $0x1d,%al
  40d2c1:	76 d8                	jbe    40d29b <__sprintf_chk@plt+0xaa0b>
  40d2c3:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40d2ca:	00 00 
  40d2cc:	eb 26                	jmp    40d2f4 <__sprintf_chk@plt+0xaa64>
  40d2ce:	66 90                	xchg   %ax,%ax
  40d2d0:	ba ff ff ff 7f       	mov    $0x7fffffff,%edx
  40d2d5:	44 29 e2             	sub    %r12d,%edx
  40d2d8:	39 d0                	cmp    %edx,%eax
  40d2da:	0f 8f e0 00 00 00    	jg     40d3c0 <__sprintf_chk@plt+0xab30>
  40d2e0:	41 01 c4             	add    %eax,%r12d
  40d2e3:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  40d2e8:	48 01 dd             	add    %rbx,%rbp
  40d2eb:	e8 40 55 ff ff       	callq  402830 <mbsinit@plt>
  40d2f0:	85 c0                	test   %eax,%eax
  40d2f2:	75 af                	jne    40d2a3 <__sprintf_chk@plt+0xaa13>
  40d2f4:	4c 89 ea             	mov    %r13,%rdx
  40d2f7:	48 8d 4c 24 10       	lea    0x10(%rsp),%rcx
  40d2fc:	48 8d 7c 24 0c       	lea    0xc(%rsp),%rdi
  40d301:	48 29 ea             	sub    %rbp,%rdx
  40d304:	48 89 ee             	mov    %rbp,%rsi
  40d307:	e8 b4 50 ff ff       	callq  4023c0 <mbrtowc@plt>
  40d30c:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  40d310:	48 89 c3             	mov    %rax,%rbx
  40d313:	74 4b                	je     40d360 <__sprintf_chk@plt+0xaad0>
  40d315:	48 83 f8 fe          	cmp    $0xfffffffffffffffe,%rax
  40d319:	0f 84 c9 00 00 00    	je     40d3e8 <__sprintf_chk@plt+0xab58>
  40d31f:	8b 7c 24 0c          	mov    0xc(%rsp),%edi
  40d323:	48 85 c0             	test   %rax,%rax
  40d326:	b8 01 00 00 00       	mov    $0x1,%eax
  40d32b:	48 0f 44 d8          	cmove  %rax,%rbx
  40d32f:	e8 fc 52 ff ff       	callq  402630 <wcwidth@plt>
  40d334:	85 c0                	test   %eax,%eax
  40d336:	79 98                	jns    40d2d0 <__sprintf_chk@plt+0xaa40>
  40d338:	45 85 f6             	test   %r14d,%r14d
  40d33b:	75 2c                	jne    40d369 <__sprintf_chk@plt+0xaad9>
  40d33d:	8b 7c 24 0c          	mov    0xc(%rsp),%edi
  40d341:	e8 5a 4f ff ff       	callq  4022a0 <iswcntrl@plt>
  40d346:	85 c0                	test   %eax,%eax
  40d348:	75 99                	jne    40d2e3 <__sprintf_chk@plt+0xaa53>
  40d34a:	41 81 fc ff ff ff 7f 	cmp    $0x7fffffff,%r12d
  40d351:	74 6d                	je     40d3c0 <__sprintf_chk@plt+0xab30>
  40d353:	41 83 c4 01          	add    $0x1,%r12d
  40d357:	eb 8a                	jmp    40d2e3 <__sprintf_chk@plt+0xaa53>
  40d359:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40d360:	45 85 ff             	test   %r15d,%r15d
  40d363:	0f 84 32 ff ff ff    	je     40d29b <__sprintf_chk@plt+0xaa0b>
  40d369:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40d36e:	eb 55                	jmp    40d3c5 <__sprintf_chk@plt+0xab35>
  40d370:	4c 39 ed             	cmp    %r13,%rbp
  40d373:	0f 83 8d 00 00 00    	jae    40d406 <__sprintf_chk@plt+0xab76>
  40d379:	e8 02 55 ff ff       	callq  402880 <__ctype_b_loc@plt>
  40d37e:	44 89 fe             	mov    %r15d,%esi
  40d381:	48 8b 08             	mov    (%rax),%rcx
  40d384:	31 c0                	xor    %eax,%eax
  40d386:	83 e6 02             	and    $0x2,%esi
  40d389:	eb 0d                	jmp    40d398 <__sprintf_chk@plt+0xab08>
  40d38b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40d390:	83 c0 01             	add    $0x1,%eax
  40d393:	4c 39 ed             	cmp    %r13,%rbp
  40d396:	74 2d                	je     40d3c5 <__sprintf_chk@plt+0xab35>
  40d398:	48 83 c5 01          	add    $0x1,%rbp
  40d39c:	0f b6 55 ff          	movzbl -0x1(%rbp),%edx
  40d3a0:	0f b7 14 51          	movzwl (%rcx,%rdx,2),%edx
  40d3a4:	f6 c6 40             	test   $0x40,%dh
  40d3a7:	75 09                	jne    40d3b2 <__sprintf_chk@plt+0xab22>
  40d3a9:	85 f6                	test   %esi,%esi
  40d3ab:	75 bc                	jne    40d369 <__sprintf_chk@plt+0xaad9>
  40d3ad:	83 e2 02             	and    $0x2,%edx
  40d3b0:	75 e1                	jne    40d393 <__sprintf_chk@plt+0xab03>
  40d3b2:	3d ff ff ff 7f       	cmp    $0x7fffffff,%eax
  40d3b7:	75 d7                	jne    40d390 <__sprintf_chk@plt+0xab00>
  40d3b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40d3c0:	b8 ff ff ff 7f       	mov    $0x7fffffff,%eax
  40d3c5:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  40d3ca:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  40d3d1:	00 00 
  40d3d3:	75 35                	jne    40d40a <__sprintf_chk@plt+0xab7a>
  40d3d5:	48 83 c4 28          	add    $0x28,%rsp
  40d3d9:	5b                   	pop    %rbx
  40d3da:	5d                   	pop    %rbp
  40d3db:	41 5c                	pop    %r12
  40d3dd:	41 5d                	pop    %r13
  40d3df:	41 5e                	pop    %r14
  40d3e1:	41 5f                	pop    %r15
  40d3e3:	c3                   	retq   
  40d3e4:	0f 1f 40 00          	nopl   0x0(%rax)
  40d3e8:	45 85 ff             	test   %r15d,%r15d
  40d3eb:	0f 85 78 ff ff ff    	jne    40d369 <__sprintf_chk@plt+0xaad9>
  40d3f1:	4c 89 ed             	mov    %r13,%rbp
  40d3f4:	41 83 c4 01          	add    $0x1,%r12d
  40d3f8:	49 39 ed             	cmp    %rbp,%r13
  40d3fb:	0f 87 ab fe ff ff    	ja     40d2ac <__sprintf_chk@plt+0xaa1c>
  40d401:	44 89 e0             	mov    %r12d,%eax
  40d404:	eb bf                	jmp    40d3c5 <__sprintf_chk@plt+0xab35>
  40d406:	31 c0                	xor    %eax,%eax
  40d408:	eb bb                	jmp    40d3c5 <__sprintf_chk@plt+0xab35>
  40d40a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40d410:	e8 8b 4f ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  40d415:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40d41c:	00 00 00 00 
  40d420:	55                   	push   %rbp
  40d421:	89 f5                	mov    %esi,%ebp
  40d423:	53                   	push   %rbx
  40d424:	48 89 fb             	mov    %rdi,%rbx
  40d427:	48 83 ec 08          	sub    $0x8,%rsp
  40d42b:	e8 50 4f ff ff       	callq  402380 <strlen@plt>
  40d430:	48 83 c4 08          	add    $0x8,%rsp
  40d434:	48 89 df             	mov    %rbx,%rdi
  40d437:	89 ea                	mov    %ebp,%edx
  40d439:	5b                   	pop    %rbx
  40d43a:	5d                   	pop    %rbp
  40d43b:	48 89 c6             	mov    %rax,%rsi
  40d43e:	e9 fd fd ff ff       	jmpq   40d240 <__sprintf_chk@plt+0xa9b0>
  40d443:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40d44a:	00 00 00 
  40d44d:	0f 1f 00             	nopl   (%rax)
  40d450:	41 57                	push   %r15
  40d452:	41 56                	push   %r14
  40d454:	41 55                	push   %r13
  40d456:	41 54                	push   %r12
  40d458:	55                   	push   %rbp
  40d459:	48 89 cd             	mov    %rcx,%rbp
  40d45c:	53                   	push   %rbx
  40d45d:	48 89 fb             	mov    %rdi,%rbx
  40d460:	48 83 ec 38          	sub    $0x38,%rsp
  40d464:	48 83 fe 02          	cmp    $0x2,%rsi
  40d468:	48 89 34 24          	mov    %rsi,(%rsp)
  40d46c:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  40d471:	77 1d                	ja     40d490 <__sprintf_chk@plt+0xac00>
  40d473:	0f 84 e7 01 00 00    	je     40d660 <__sprintf_chk@plt+0xadd0>
  40d479:	48 83 c4 38          	add    $0x38,%rsp
  40d47d:	5b                   	pop    %rbx
  40d47e:	5d                   	pop    %rbp
  40d47f:	41 5c                	pop    %r12
  40d481:	41 5d                	pop    %r13
  40d483:	41 5e                	pop    %r14
  40d485:	41 5f                	pop    %r15
  40d487:	c3                   	retq   
  40d488:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40d48f:	00 
  40d490:	48 8b 34 24          	mov    (%rsp),%rsi
  40d494:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  40d499:	48 89 f0             	mov    %rsi,%rax
  40d49c:	48 d1 e8             	shr    %rax
  40d49f:	49 89 c7             	mov    %rax,%r15
  40d4a2:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40d4a7:	48 8d 04 c7          	lea    (%rdi,%rax,8),%rax
  40d4ab:	4c 29 fe             	sub    %r15,%rsi
  40d4ae:	48 89 c7             	mov    %rax,%rdi
  40d4b1:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40d4b6:	e8 95 ff ff ff       	callq  40d450 <__sprintf_chk@plt+0xabc0>
  40d4bb:	49 83 ff 01          	cmp    $0x1,%r15
  40d4bf:	0f 84 ab 00 00 00    	je     40d570 <__sprintf_chk@plt+0xace0>
  40d4c5:	48 8b 04 24          	mov    (%rsp),%rax
  40d4c9:	4c 8b 7c 24 18       	mov    0x18(%rsp),%r15
  40d4ce:	48 89 e9             	mov    %rbp,%rcx
  40d4d1:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
  40d4d6:	48 c1 e8 02          	shr    $0x2,%rax
  40d4da:	4c 89 fa             	mov    %r15,%rdx
  40d4dd:	4c 8d 2c c3          	lea    (%rbx,%rax,8),%r13
  40d4e1:	49 89 c6             	mov    %rax,%r14
  40d4e4:	48 29 c6             	sub    %rax,%rsi
  40d4e7:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40d4ec:	4c 89 ef             	mov    %r13,%rdi
  40d4ef:	e8 5c ff ff ff       	callq  40d450 <__sprintf_chk@plt+0xabc0>
  40d4f4:	4c 89 fa             	mov    %r15,%rdx
  40d4f7:	48 89 e9             	mov    %rbp,%rcx
  40d4fa:	4c 89 f6             	mov    %r14,%rsi
  40d4fd:	48 89 df             	mov    %rbx,%rdi
  40d500:	4d 8d 7f 08          	lea    0x8(%r15),%r15
  40d504:	e8 47 ff ff ff       	callq  40d450 <__sprintf_chk@plt+0xabc0>
  40d509:	4c 8b 23             	mov    (%rbx),%r12
  40d50c:	4d 8b 6d 00          	mov    0x0(%r13),%r13
  40d510:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40d517:	00 00 
  40d519:	eb 20                	jmp    40d53b <__sprintf_chk@plt+0xacab>
  40d51b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40d520:	49 83 c6 01          	add    $0x1,%r14
  40d524:	4c 39 74 24 08       	cmp    %r14,0x8(%rsp)
  40d529:	4d 89 6f f8          	mov    %r13,-0x8(%r15)
  40d52d:	0f 84 01 01 00 00    	je     40d634 <__sprintf_chk@plt+0xada4>
  40d533:	4e 8b 2c f3          	mov    (%rbx,%r14,8),%r13
  40d537:	49 83 c7 08          	add    $0x8,%r15
  40d53b:	4c 89 ee             	mov    %r13,%rsi
  40d53e:	4c 89 e7             	mov    %r12,%rdi
  40d541:	ff d5                	callq  *%rbp
  40d543:	85 c0                	test   %eax,%eax
  40d545:	7f d9                	jg     40d520 <__sprintf_chk@plt+0xac90>
  40d547:	48 83 44 24 10 01    	addq   $0x1,0x10(%rsp)
  40d54d:	4d 89 67 f8          	mov    %r12,-0x8(%r15)
  40d551:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d556:	48 39 44 24 20       	cmp    %rax,0x20(%rsp)
  40d55b:	0f 84 c4 00 00 00    	je     40d625 <__sprintf_chk@plt+0xad95>
  40d561:	4c 8b 24 c3          	mov    (%rbx,%rax,8),%r12
  40d565:	eb d0                	jmp    40d537 <__sprintf_chk@plt+0xaca7>
  40d567:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40d56e:	00 00 
  40d570:	4c 8b 23             	mov    (%rbx),%r12
  40d573:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40d578:	4c 89 20             	mov    %r12,(%rax)
  40d57b:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40d580:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  40d585:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40d58b:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
  40d592:	00 00 
  40d594:	4c 8b 28             	mov    (%rax),%r13
  40d597:	eb 1e                	jmp    40d5b7 <__sprintf_chk@plt+0xad27>
  40d599:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40d5a0:	49 83 c6 01          	add    $0x1,%r14
  40d5a4:	4c 39 34 24          	cmp    %r14,(%rsp)
  40d5a8:	4e 89 6c fb f8       	mov    %r13,-0x8(%rbx,%r15,8)
  40d5ad:	74 41                	je     40d5f0 <__sprintf_chk@plt+0xad60>
  40d5af:	4e 8b 2c f3          	mov    (%rbx,%r14,8),%r13
  40d5b3:	49 83 c7 01          	add    $0x1,%r15
  40d5b7:	4c 89 ee             	mov    %r13,%rsi
  40d5ba:	4c 89 e7             	mov    %r12,%rdi
  40d5bd:	ff d5                	callq  *%rbp
  40d5bf:	85 c0                	test   %eax,%eax
  40d5c1:	7f dd                	jg     40d5a0 <__sprintf_chk@plt+0xad10>
  40d5c3:	48 83 44 24 10 01    	addq   $0x1,0x10(%rsp)
  40d5c9:	4e 89 64 fb f8       	mov    %r12,-0x8(%rbx,%r15,8)
  40d5ce:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d5d3:	48 39 44 24 08       	cmp    %rax,0x8(%rsp)
  40d5d8:	0f 84 9b fe ff ff    	je     40d479 <__sprintf_chk@plt+0xabe9>
  40d5de:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40d5e3:	4c 8b 24 c1          	mov    (%rcx,%rax,8),%r12
  40d5e7:	eb ca                	jmp    40d5b3 <__sprintf_chk@plt+0xad23>
  40d5e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40d5f0:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d5f5:	4a 8d 3c fb          	lea    (%rbx,%r15,8),%rdi
  40d5f9:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40d5fe:	4c 8b 7c 24 08       	mov    0x8(%rsp),%r15
  40d603:	48 83 c4 38          	add    $0x38,%rsp
  40d607:	5b                   	pop    %rbx
  40d608:	5d                   	pop    %rbp
  40d609:	41 5c                	pop    %r12
  40d60b:	41 5d                	pop    %r13
  40d60d:	49 29 c7             	sub    %rax,%r15
  40d610:	48 8d 34 c1          	lea    (%rcx,%rax,8),%rsi
  40d614:	41 5e                	pop    %r14
  40d616:	4a 8d 14 fd 00 00 00 	lea    0x0(,%r15,8),%rdx
  40d61d:	00 
  40d61e:	41 5f                	pop    %r15
  40d620:	e9 9b 4f ff ff       	jmpq   4025c0 <memcpy@plt>
  40d625:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40d62a:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
  40d62f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40d634:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d639:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  40d63e:	4c 89 ff             	mov    %r15,%rdi
  40d641:	48 29 c2             	sub    %rax,%rdx
  40d644:	48 8d 34 c3          	lea    (%rbx,%rax,8),%rsi
  40d648:	48 c1 e2 03          	shl    $0x3,%rdx
  40d64c:	e8 6f 4f ff ff       	callq  4025c0 <memcpy@plt>
  40d651:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40d656:	4c 8b 20             	mov    (%rax),%r12
  40d659:	e9 1d ff ff ff       	jmpq   40d57b <__sprintf_chk@plt+0xaceb>
  40d65e:	66 90                	xchg   %ax,%ax
  40d660:	4c 8b 6f 08          	mov    0x8(%rdi),%r13
  40d664:	4c 8b 27             	mov    (%rdi),%r12
  40d667:	4c 89 ee             	mov    %r13,%rsi
  40d66a:	4c 89 e7             	mov    %r12,%rdi
  40d66d:	ff d1                	callq  *%rcx
  40d66f:	85 c0                	test   %eax,%eax
  40d671:	0f 8e 02 fe ff ff    	jle    40d479 <__sprintf_chk@plt+0xabe9>
  40d677:	4c 89 63 08          	mov    %r12,0x8(%rbx)
  40d67b:	4c 89 2b             	mov    %r13,(%rbx)
  40d67e:	48 83 c4 38          	add    $0x38,%rsp
  40d682:	5b                   	pop    %rbx
  40d683:	5d                   	pop    %rbp
  40d684:	41 5c                	pop    %r12
  40d686:	41 5d                	pop    %r13
  40d688:	41 5e                	pop    %r14
  40d68a:	41 5f                	pop    %r15
  40d68c:	c3                   	retq   
  40d68d:	0f 1f 00             	nopl   (%rax)
  40d690:	48 8d 04 f7          	lea    (%rdi,%rsi,8),%rax
  40d694:	48 89 d1             	mov    %rdx,%rcx
  40d697:	48 89 c2             	mov    %rax,%rdx
  40d69a:	e9 b1 fd ff ff       	jmpq   40d450 <__sprintf_chk@plt+0xabc0>
  40d69f:	90                   	nop
  40d6a0:	48 85 ff             	test   %rdi,%rdi
  40d6a3:	53                   	push   %rbx
  40d6a4:	48 89 fb             	mov    %rdi,%rbx
  40d6a7:	74 6a                	je     40d713 <__sprintf_chk@plt+0xae83>
  40d6a9:	be 2f 00 00 00       	mov    $0x2f,%esi
  40d6ae:	e8 5d 4d ff ff       	callq  402410 <strrchr@plt>
  40d6b3:	48 85 c0             	test   %rax,%rax
  40d6b6:	74 4b                	je     40d703 <__sprintf_chk@plt+0xae73>
  40d6b8:	48 8d 50 01          	lea    0x1(%rax),%rdx
  40d6bc:	48 89 d1             	mov    %rdx,%rcx
  40d6bf:	48 29 d9             	sub    %rbx,%rcx
  40d6c2:	48 83 f9 06          	cmp    $0x6,%rcx
  40d6c6:	7e 3b                	jle    40d703 <__sprintf_chk@plt+0xae73>
  40d6c8:	48 8d 70 fa          	lea    -0x6(%rax),%rsi
  40d6cc:	bf d8 5f 41 00       	mov    $0x415fd8,%edi
  40d6d1:	b9 07 00 00 00       	mov    $0x7,%ecx
  40d6d6:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  40d6d8:	75 29                	jne    40d703 <__sprintf_chk@plt+0xae73>
  40d6da:	b9 03 00 00 00       	mov    $0x3,%ecx
  40d6df:	48 89 d6             	mov    %rdx,%rsi
  40d6e2:	bf e0 5f 41 00       	mov    $0x415fe0,%edi
  40d6e7:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  40d6e9:	48 89 d3             	mov    %rdx,%rbx
  40d6ec:	40 0f 97 c6          	seta   %sil
  40d6f0:	0f 92 c1             	setb   %cl
  40d6f3:	40 38 ce             	cmp    %cl,%sil
  40d6f6:	75 0b                	jne    40d703 <__sprintf_chk@plt+0xae73>
  40d6f8:	48 8d 58 04          	lea    0x4(%rax),%rbx
  40d6fc:	48 89 1d fd ce 20 00 	mov    %rbx,0x20cefd(%rip)        # 61a600 <__progname@@GLIBC_2.2.5>
  40d703:	48 89 1d f6 da 20 00 	mov    %rbx,0x20daf6(%rip)        # 61b200 <stderr@@GLIBC_2.2.5+0xbb0>
  40d70a:	48 89 1d 37 cf 20 00 	mov    %rbx,0x20cf37(%rip)        # 61a648 <__progname_full@@GLIBC_2.2.5>
  40d711:	5b                   	pop    %rbx
  40d712:	c3                   	retq   
  40d713:	48 8b 0d 36 cf 20 00 	mov    0x20cf36(%rip),%rcx        # 61a650 <stderr@@GLIBC_2.2.5>
  40d71a:	ba 37 00 00 00       	mov    $0x37,%edx
  40d71f:	be 01 00 00 00       	mov    $0x1,%esi
  40d724:	bf a0 5f 41 00       	mov    $0x415fa0,%edi
  40d729:	e8 d2 50 ff ff       	callq  402800 <fwrite@plt>
  40d72e:	e8 ed 4a ff ff       	callq  402220 <abort@plt>
  40d733:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40d73a:	00 00 00 
  40d73d:	0f 1f 00             	nopl   (%rax)
  40d740:	48 83 ec 48          	sub    $0x48,%rsp
  40d744:	31 c0                	xor    %eax,%eax
  40d746:	48 89 fa             	mov    %rdi,%rdx
  40d749:	b9 07 00 00 00       	mov    $0x7,%ecx
  40d74e:	48 89 e7             	mov    %rsp,%rdi
  40d751:	83 fe 08             	cmp    $0x8,%esi
  40d754:	f3 48 ab             	rep stos %rax,%es:(%rdi)
  40d757:	74 48                	je     40d7a1 <__sprintf_chk@plt+0xaf11>
  40d759:	89 34 24             	mov    %esi,(%rsp)
  40d75c:	48 8b 04 24          	mov    (%rsp),%rax
  40d760:	48 89 02             	mov    %rax,(%rdx)
  40d763:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  40d768:	48 89 42 08          	mov    %rax,0x8(%rdx)
  40d76c:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40d771:	48 89 42 10          	mov    %rax,0x10(%rdx)
  40d775:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40d77a:	48 89 42 18          	mov    %rax,0x18(%rdx)
  40d77e:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  40d783:	48 89 42 20          	mov    %rax,0x20(%rdx)
  40d787:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40d78c:	48 89 42 28          	mov    %rax,0x28(%rdx)
  40d790:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  40d795:	48 89 42 30          	mov    %rax,0x30(%rdx)
  40d799:	48 89 d0             	mov    %rdx,%rax
  40d79c:	48 83 c4 48          	add    $0x48,%rsp
  40d7a0:	c3                   	retq   
  40d7a1:	e8 7a 4a ff ff       	callq  402220 <abort@plt>
  40d7a6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40d7ad:	00 00 00 
  40d7b0:	41 55                	push   %r13
  40d7b2:	ba 05 00 00 00       	mov    $0x5,%edx
  40d7b7:	41 54                	push   %r12
  40d7b9:	41 89 f4             	mov    %esi,%r12d
  40d7bc:	48 89 fe             	mov    %rdi,%rsi
  40d7bf:	55                   	push   %rbp
  40d7c0:	48 89 fd             	mov    %rdi,%rbp
  40d7c3:	31 ff                	xor    %edi,%edi
  40d7c5:	53                   	push   %rbx
  40d7c6:	48 83 ec 08          	sub    $0x8,%rsp
  40d7ca:	e8 91 4b ff ff       	callq  402360 <dcgettext@plt>
  40d7cf:	48 39 e8             	cmp    %rbp,%rax
  40d7d2:	48 89 c3             	mov    %rax,%rbx
  40d7d5:	74 11                	je     40d7e8 <__sprintf_chk@plt+0xaf58>
  40d7d7:	48 83 c4 08          	add    $0x8,%rsp
  40d7db:	48 89 d8             	mov    %rbx,%rax
  40d7de:	5b                   	pop    %rbx
  40d7df:	5d                   	pop    %rbp
  40d7e0:	41 5c                	pop    %r12
  40d7e2:	41 5d                	pop    %r13
  40d7e4:	c3                   	retq   
  40d7e5:	0f 1f 00             	nopl   (%rax)
  40d7e8:	e8 13 41 00 00       	callq  411900 <__sprintf_chk@plt+0xf070>
  40d7ed:	0f b6 10             	movzbl (%rax),%edx
  40d7f0:	83 e2 df             	and    $0xffffffdf,%edx
  40d7f3:	80 fa 55             	cmp    $0x55,%dl
  40d7f6:	75 58                	jne    40d850 <__sprintf_chk@plt+0xafc0>
  40d7f8:	0f b6 50 01          	movzbl 0x1(%rax),%edx
  40d7fc:	83 e2 df             	and    $0xffffffdf,%edx
  40d7ff:	80 fa 54             	cmp    $0x54,%dl
  40d802:	75 34                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d804:	0f b6 50 02          	movzbl 0x2(%rax),%edx
  40d808:	83 e2 df             	and    $0xffffffdf,%edx
  40d80b:	80 fa 46             	cmp    $0x46,%dl
  40d80e:	75 28                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d810:	80 78 03 2d          	cmpb   $0x2d,0x3(%rax)
  40d814:	75 22                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d816:	80 78 04 38          	cmpb   $0x38,0x4(%rax)
  40d81a:	75 1c                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d81c:	80 78 05 00          	cmpb   $0x0,0x5(%rax)
  40d820:	75 16                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d822:	80 3b 60             	cmpb   $0x60,(%rbx)
  40d825:	b8 f1 5f 41 00       	mov    $0x415ff1,%eax
  40d82a:	bb e4 5f 41 00       	mov    $0x415fe4,%ebx
  40d82f:	48 0f 44 d8          	cmove  %rax,%rbx
  40d833:	eb a2                	jmp    40d7d7 <__sprintf_chk@plt+0xaf47>
  40d835:	0f 1f 00             	nopl   (%rax)
  40d838:	bb eb 5f 41 00       	mov    $0x415feb,%ebx
  40d83d:	41 83 fc 07          	cmp    $0x7,%r12d
  40d841:	b8 ea 6d 41 00       	mov    $0x416dea,%eax
  40d846:	48 0f 45 d8          	cmovne %rax,%rbx
  40d84a:	eb 8b                	jmp    40d7d7 <__sprintf_chk@plt+0xaf47>
  40d84c:	0f 1f 40 00          	nopl   0x0(%rax)
  40d850:	80 fa 47             	cmp    $0x47,%dl
  40d853:	75 e3                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d855:	0f b6 50 01          	movzbl 0x1(%rax),%edx
  40d859:	83 e2 df             	and    $0xffffffdf,%edx
  40d85c:	80 fa 42             	cmp    $0x42,%dl
  40d85f:	75 d7                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d861:	80 78 02 31          	cmpb   $0x31,0x2(%rax)
  40d865:	75 d1                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d867:	80 78 03 38          	cmpb   $0x38,0x3(%rax)
  40d86b:	75 cb                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d86d:	80 78 04 30          	cmpb   $0x30,0x4(%rax)
  40d871:	75 c5                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d873:	80 78 05 33          	cmpb   $0x33,0x5(%rax)
  40d877:	75 bf                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d879:	80 78 06 30          	cmpb   $0x30,0x6(%rax)
  40d87d:	75 b9                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d87f:	80 78 07 00          	cmpb   $0x0,0x7(%rax)
  40d883:	75 b3                	jne    40d838 <__sprintf_chk@plt+0xafa8>
  40d885:	49 89 dd             	mov    %rbx,%r13
  40d888:	b8 e8 5f 41 00       	mov    $0x415fe8,%eax
  40d88d:	bb ed 5f 41 00       	mov    $0x415fed,%ebx
  40d892:	41 80 7d 00 60       	cmpb   $0x60,0x0(%r13)
  40d897:	48 0f 45 d8          	cmovne %rax,%rbx
  40d89b:	e9 37 ff ff ff       	jmpq   40d7d7 <__sprintf_chk@plt+0xaf47>
  40d8a0:	41 57                	push   %r15
  40d8a2:	49 89 cf             	mov    %rcx,%r15
  40d8a5:	41 56                	push   %r14
  40d8a7:	45 89 c6             	mov    %r8d,%r14d
  40d8aa:	41 55                	push   %r13
  40d8ac:	49 89 d5             	mov    %rdx,%r13
  40d8af:	41 54                	push   %r12
  40d8b1:	55                   	push   %rbp
  40d8b2:	53                   	push   %rbx
  40d8b3:	44 89 cb             	mov    %r9d,%ebx
  40d8b6:	48 81 ec c8 00 00 00 	sub    $0xc8,%rsp
  40d8bd:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  40d8c4:	00 
  40d8c5:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
  40d8ca:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  40d8cf:	44 89 44 24 34       	mov    %r8d,0x34(%rsp)
  40d8d4:	44 89 8c 24 90 00 00 	mov    %r9d,0x90(%rsp)
  40d8db:	00 
  40d8dc:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
  40d8e1:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  40d8e8:	00 
  40d8e9:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  40d8ee:	48 8b 84 24 10 01 00 	mov    0x110(%rsp),%rax
  40d8f5:	00 
  40d8f6:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40d8fb:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  40d902:	00 00 
  40d904:	48 89 84 24 b8 00 00 	mov    %rax,0xb8(%rsp)
  40d90b:	00 
  40d90c:	31 c0                	xor    %eax,%eax
  40d90e:	e8 5d 4a ff ff       	callq  402370 <__ctype_get_mb_cur_max@plt>
  40d913:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  40d918:	89 d8                	mov    %ebx,%eax
  40d91a:	d1 e8                	shr    %eax
  40d91c:	83 e0 01             	and    $0x1,%eax
  40d91f:	41 83 fe 08          	cmp    $0x8,%r14d
  40d923:	88 44 24 33          	mov    %al,0x33(%rsp)
  40d927:	0f 87 53 09 00 00    	ja     40e280 <__sprintf_chk@plt+0xb9f0>
  40d92d:	44 89 f0             	mov    %r14d,%eax
  40d930:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
  40d935:	ff 24 c5 20 60 41 00 	jmpq   *0x416020(,%rax,8)
  40d93c:	0f 1f 40 00          	nopl   0x0(%rax)
  40d940:	c6 44 24 33 00       	movb   $0x0,0x33(%rsp)
  40d945:	c6 44 24 20 00       	movb   $0x0,0x20(%rsp)
  40d94a:	45 31 f6             	xor    %r14d,%r14d
  40d94d:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  40d954:	00 00 
  40d956:	31 db                	xor    %ebx,%ebx
  40d958:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40d95f:	00 
  40d960:	0f b6 44 24 33       	movzbl 0x33(%rsp),%eax
  40d965:	4d 89 f1             	mov    %r14,%r9
  40d968:	31 ed                	xor    %ebp,%ebp
  40d96a:	4d 89 de             	mov    %r11,%r14
  40d96d:	4d 89 e8             	mov    %r13,%r8
  40d970:	83 f0 01             	xor    $0x1,%eax
  40d973:	88 44 24 38          	mov    %al,0x38(%rsp)
  40d977:	0f b6 44 24 20       	movzbl 0x20(%rsp),%eax
  40d97c:	83 f0 01             	xor    $0x1,%eax
  40d97f:	88 84 24 95 00 00 00 	mov    %al,0x95(%rsp)
  40d986:	4c 39 fd             	cmp    %r15,%rbp
  40d989:	0f 95 c0             	setne  %al
  40d98c:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
  40d990:	0f 84 e0 01 00 00    	je     40db76 <__sprintf_chk@plt+0xb2e6>
  40d996:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40d99d:	00 00 00 
  40d9a0:	84 c0                	test   %al,%al
  40d9a2:	0f 84 de 01 00 00    	je     40db86 <__sprintf_chk@plt+0xb2f6>
  40d9a8:	4d 85 c9             	test   %r9,%r9
  40d9ab:	0f 95 c1             	setne  %cl
  40d9ae:	0f 84 ec 06 00 00    	je     40e0a0 <__sprintf_chk@plt+0xb810>
  40d9b4:	80 7c 24 20 00       	cmpb   $0x0,0x20(%rsp)
  40d9b9:	0f 84 e1 06 00 00    	je     40e0a0 <__sprintf_chk@plt+0xb810>
  40d9bf:	4a 8d 44 0d 00       	lea    0x0(%rbp,%r9,1),%rax
  40d9c4:	49 39 c7             	cmp    %rax,%r15
  40d9c7:	0f 82 d3 06 00 00    	jb     40e0a0 <__sprintf_chk@plt+0xb810>
  40d9cd:	4d 8d 2c 28          	lea    (%r8,%rbp,1),%r13
  40d9d1:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
  40d9d6:	4c 89 ca             	mov    %r9,%rdx
  40d9d9:	89 4c 24 50          	mov    %ecx,0x50(%rsp)
  40d9dd:	4c 89 44 24 48       	mov    %r8,0x48(%rsp)
  40d9e2:	4c 89 ef             	mov    %r13,%rdi
  40d9e5:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
  40d9ea:	e8 11 4b ff ff       	callq  402500 <memcmp@plt>
  40d9ef:	85 c0                	test   %eax,%eax
  40d9f1:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
  40d9f6:	4c 8b 44 24 48       	mov    0x48(%rsp),%r8
  40d9fb:	8b 4c 24 50          	mov    0x50(%rsp),%ecx
  40d9ff:	0f 85 ab 06 00 00    	jne    40e0b0 <__sprintf_chk@plt+0xb820>
  40da05:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40da0a:	0f 85 30 02 00 00    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40da10:	41 bb 01 00 00 00    	mov    $0x1,%r11d
  40da16:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40da1d:	00 00 00 
  40da20:	45 0f b6 65 00       	movzbl 0x0(%r13),%r12d
  40da25:	41 80 fc 7e          	cmp    $0x7e,%r12b
  40da29:	0f 87 e9 03 00 00    	ja     40de18 <__sprintf_chk@plt+0xb588>
  40da2f:	41 0f b6 c4          	movzbl %r12b,%eax
  40da33:	ff 24 c5 68 60 41 00 	jmpq   *0x416068(,%rax,8)
  40da3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40da40:	8b 44 24 34          	mov    0x34(%rsp),%eax
  40da44:	83 f8 02             	cmp    $0x2,%eax
  40da47:	0f 84 e3 01 00 00    	je     40dc30 <__sprintf_chk@plt+0xb3a0>
  40da4d:	83 f8 03             	cmp    $0x3,%eax
  40da50:	0f 85 a2 00 00 00    	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40da56:	f6 84 24 90 00 00 00 	testb  $0x4,0x90(%rsp)
  40da5d:	04 
  40da5e:	0f 84 94 00 00 00    	je     40daf8 <__sprintf_chk@plt+0xb268>
  40da64:	48 8d 45 02          	lea    0x2(%rbp),%rax
  40da68:	49 39 c7             	cmp    %rax,%r15
  40da6b:	0f 86 87 00 00 00    	jbe    40daf8 <__sprintf_chk@plt+0xb268>
  40da71:	41 80 7c 28 01 3f    	cmpb   $0x3f,0x1(%r8,%rbp,1)
  40da77:	75 7f                	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40da79:	41 0f b6 34 00       	movzbl (%r8,%rax,1),%esi
  40da7e:	8d 4e df             	lea    -0x21(%rsi),%ecx
  40da81:	80 f9 1d             	cmp    $0x1d,%cl
  40da84:	77 72                	ja     40daf8 <__sprintf_chk@plt+0xb268>
  40da86:	ba 01 00 00 00       	mov    $0x1,%edx
  40da8b:	48 d3 e2             	shl    %cl,%rdx
  40da8e:	f7 c2 c1 51 00 38    	test   $0x380051c1,%edx
  40da94:	74 62                	je     40daf8 <__sprintf_chk@plt+0xb268>
  40da96:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40da9b:	0f 85 9f 01 00 00    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40daa1:	4c 39 f3             	cmp    %r14,%rbx
  40daa4:	73 09                	jae    40daaf <__sprintf_chk@plt+0xb21f>
  40daa6:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  40daab:	c6 04 1f 3f          	movb   $0x3f,(%rdi,%rbx,1)
  40daaf:	48 8d 53 01          	lea    0x1(%rbx),%rdx
  40dab3:	49 39 d6             	cmp    %rdx,%r14
  40dab6:	76 0a                	jbe    40dac2 <__sprintf_chk@plt+0xb232>
  40dab8:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  40dabd:	c6 44 1f 01 22       	movb   $0x22,0x1(%rdi,%rbx,1)
  40dac2:	48 8d 53 02          	lea    0x2(%rbx),%rdx
  40dac6:	49 39 d6             	cmp    %rdx,%r14
  40dac9:	76 0a                	jbe    40dad5 <__sprintf_chk@plt+0xb245>
  40dacb:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  40dad0:	c6 44 1f 02 22       	movb   $0x22,0x2(%rdi,%rbx,1)
  40dad5:	48 8d 53 03          	lea    0x3(%rbx),%rdx
  40dad9:	49 39 d6             	cmp    %rdx,%r14
  40dadc:	76 0a                	jbe    40dae8 <__sprintf_chk@plt+0xb258>
  40dade:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  40dae3:	c6 44 1f 03 3f       	movb   $0x3f,0x3(%rdi,%rbx,1)
  40dae8:	48 83 c3 04          	add    $0x4,%rbx
  40daec:	41 89 f4             	mov    %esi,%r12d
  40daef:	48 89 c5             	mov    %rax,%rbp
  40daf2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40daf8:	80 7c 24 38 00       	cmpb   $0x0,0x38(%rsp)
  40dafd:	74 0a                	je     40db09 <__sprintf_chk@plt+0xb279>
  40daff:	80 bc 24 95 00 00 00 	cmpb   $0x0,0x95(%rsp)
  40db06:	00 
  40db07:	75 25                	jne    40db2e <__sprintf_chk@plt+0xb29e>
  40db09:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  40db0e:	48 85 ff             	test   %rdi,%rdi
  40db11:	74 1b                	je     40db2e <__sprintf_chk@plt+0xb29e>
  40db13:	44 89 e2             	mov    %r12d,%edx
  40db16:	44 89 e1             	mov    %r12d,%ecx
  40db19:	b8 01 00 00 00       	mov    $0x1,%eax
  40db1e:	c0 ea 05             	shr    $0x5,%dl
  40db21:	83 e1 1f             	and    $0x1f,%ecx
  40db24:	0f b6 d2             	movzbl %dl,%edx
  40db27:	d3 e0                	shl    %cl,%eax
  40db29:	85 04 97             	test   %eax,(%rdi,%rdx,4)
  40db2c:	75 05                	jne    40db33 <__sprintf_chk@plt+0xb2a3>
  40db2e:	45 84 db             	test   %r11b,%r11b
  40db31:	74 1d                	je     40db50 <__sprintf_chk@plt+0xb2c0>
  40db33:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40db38:	0f 85 02 01 00 00    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40db3e:	4c 39 f3             	cmp    %r14,%rbx
  40db41:	73 09                	jae    40db4c <__sprintf_chk@plt+0xb2bc>
  40db43:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40db48:	c6 04 18 5c          	movb   $0x5c,(%rax,%rbx,1)
  40db4c:	48 83 c3 01          	add    $0x1,%rbx
  40db50:	48 83 c5 01          	add    $0x1,%rbp
  40db54:	4c 39 f3             	cmp    %r14,%rbx
  40db57:	73 09                	jae    40db62 <__sprintf_chk@plt+0xb2d2>
  40db59:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40db5e:	44 88 24 18          	mov    %r12b,(%rax,%rbx,1)
  40db62:	48 83 c3 01          	add    $0x1,%rbx
  40db66:	4c 39 fd             	cmp    %r15,%rbp
  40db69:	0f 95 c0             	setne  %al
  40db6c:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
  40db70:	0f 85 2a fe ff ff    	jne    40d9a0 <__sprintf_chk@plt+0xb110>
  40db76:	41 80 3c 28 00       	cmpb   $0x0,(%r8,%rbp,1)
  40db7b:	0f 95 c0             	setne  %al
  40db7e:	84 c0                	test   %al,%al
  40db80:	0f 85 22 fe ff ff    	jne    40d9a8 <__sprintf_chk@plt+0xb118>
  40db86:	48 85 db             	test   %rbx,%rbx
  40db89:	4d 89 f3             	mov    %r14,%r11
  40db8c:	4d 89 c5             	mov    %r8,%r13
  40db8f:	75 12                	jne    40dba3 <__sprintf_chk@plt+0xb313>
  40db91:	83 7c 24 34 02       	cmpl   $0x2,0x34(%rsp)
  40db96:	75 0b                	jne    40dba3 <__sprintf_chk@plt+0xb313>
  40db98:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40db9d:	0f 85 a3 00 00 00    	jne    40dc46 <__sprintf_chk@plt+0xb3b6>
  40dba3:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40dba8:	75 3a                	jne    40dbe4 <__sprintf_chk@plt+0xb354>
  40dbaa:	48 83 7c 24 60 00    	cmpq   $0x0,0x60(%rsp)
  40dbb0:	74 32                	je     40dbe4 <__sprintf_chk@plt+0xb354>
  40dbb2:	48 8b 54 24 60       	mov    0x60(%rsp),%rdx
  40dbb7:	0f b6 02             	movzbl (%rdx),%eax
  40dbba:	84 c0                	test   %al,%al
  40dbbc:	74 26                	je     40dbe4 <__sprintf_chk@plt+0xb354>
  40dbbe:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40dbc3:	48 29 da             	sub    %rbx,%rdx
  40dbc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40dbcd:	00 00 00 
  40dbd0:	49 39 db             	cmp    %rbx,%r11
  40dbd3:	76 03                	jbe    40dbd8 <__sprintf_chk@plt+0xb348>
  40dbd5:	88 04 19             	mov    %al,(%rcx,%rbx,1)
  40dbd8:	48 83 c3 01          	add    $0x1,%rbx
  40dbdc:	0f b6 04 1a          	movzbl (%rdx,%rbx,1),%eax
  40dbe0:	84 c0                	test   %al,%al
  40dbe2:	75 ec                	jne    40dbd0 <__sprintf_chk@plt+0xb340>
  40dbe4:	4c 39 db             	cmp    %r11,%rbx
  40dbe7:	48 89 d8             	mov    %rbx,%rax
  40dbea:	0f 83 96 00 00 00    	jae    40dc86 <__sprintf_chk@plt+0xb3f6>
  40dbf0:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  40dbf5:	c6 04 1e 00          	movb   $0x0,(%rsi,%rbx,1)
  40dbf9:	e9 88 00 00 00       	jmpq   40dc86 <__sprintf_chk@plt+0xb3f6>
  40dbfe:	66 90                	xchg   %ax,%ax
  40dc00:	49 83 ff 01          	cmp    $0x1,%r15
  40dc04:	0f 95 c0             	setne  %al
  40dc07:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
  40dc0b:	0f 84 5f 06 00 00    	je     40e270 <__sprintf_chk@plt+0xb9e0>
  40dc11:	84 c0                	test   %al,%al
  40dc13:	0f 85 df fe ff ff    	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40dc19:	48 85 ed             	test   %rbp,%rbp
  40dc1c:	0f 85 d6 fe ff ff    	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40dc22:	83 7c 24 34 02       	cmpl   $0x2,0x34(%rsp)
  40dc27:	0f 85 cb fe ff ff    	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40dc2d:	0f 1f 00             	nopl   (%rax)
  40dc30:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40dc35:	0f 84 bd fe ff ff    	je     40daf8 <__sprintf_chk@plt+0xb268>
  40dc3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40dc40:	4d 89 f3             	mov    %r14,%r11
  40dc43:	4d 89 c5             	mov    %r8,%r13
  40dc46:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  40dc4b:	44 8b 8c 24 90 00 00 	mov    0x90(%rsp),%r9d
  40dc52:	00 
  40dc53:	4c 89 f9             	mov    %r15,%rcx
  40dc56:	44 8b 44 24 34       	mov    0x34(%rsp),%r8d
  40dc5b:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
  40dc60:	4c 89 ea             	mov    %r13,%rdx
  40dc63:	48 c7 04 24 00 00 00 	movq   $0x0,(%rsp)
  40dc6a:	00 
  40dc6b:	4c 89 de             	mov    %r11,%rsi
  40dc6e:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40dc73:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  40dc78:	41 83 e1 fd          	and    $0xfffffffd,%r9d
  40dc7c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40dc81:	e8 1a fc ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40dc86:	48 8b b4 24 b8 00 00 	mov    0xb8(%rsp),%rsi
  40dc8d:	00 
  40dc8e:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  40dc95:	00 00 
  40dc97:	0f 85 86 07 00 00    	jne    40e423 <__sprintf_chk@plt+0xbb93>
  40dc9d:	48 81 c4 c8 00 00 00 	add    $0xc8,%rsp
  40dca4:	5b                   	pop    %rbx
  40dca5:	5d                   	pop    %rbp
  40dca6:	41 5c                	pop    %r12
  40dca8:	41 5d                	pop    %r13
  40dcaa:	41 5e                	pop    %r14
  40dcac:	41 5f                	pop    %r15
  40dcae:	c3                   	retq   
  40dcaf:	90                   	nop
  40dcb0:	b8 72 00 00 00       	mov    $0x72,%eax
  40dcb5:	83 7c 24 34 02       	cmpl   $0x2,0x34(%rsp)
  40dcba:	0f 84 2f 06 00 00    	je     40e2ef <__sprintf_chk@plt+0xba5f>
  40dcc0:	80 7c 24 20 00       	cmpb   $0x0,0x20(%rsp)
  40dcc5:	0f 84 2d fe ff ff    	je     40daf8 <__sprintf_chk@plt+0xb268>
  40dccb:	41 89 c4             	mov    %eax,%r12d
  40dcce:	e9 60 fe ff ff       	jmpq   40db33 <__sprintf_chk@plt+0xb2a3>
  40dcd3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40dcd8:	b8 62 00 00 00       	mov    $0x62,%eax
  40dcdd:	eb e1                	jmp    40dcc0 <__sprintf_chk@plt+0xb430>
  40dcdf:	90                   	nop
  40dce0:	b8 66 00 00 00       	mov    $0x66,%eax
  40dce5:	eb d9                	jmp    40dcc0 <__sprintf_chk@plt+0xb430>
  40dce7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40dcee:	00 00 
  40dcf0:	b8 76 00 00 00       	mov    $0x76,%eax
  40dcf5:	eb c9                	jmp    40dcc0 <__sprintf_chk@plt+0xb430>
  40dcf7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40dcfe:	00 00 
  40dd00:	b8 6e 00 00 00       	mov    $0x6e,%eax
  40dd05:	eb ae                	jmp    40dcb5 <__sprintf_chk@plt+0xb425>
  40dd07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40dd0e:	00 00 
  40dd10:	b8 74 00 00 00       	mov    $0x74,%eax
  40dd15:	eb 9e                	jmp    40dcb5 <__sprintf_chk@plt+0xb425>
  40dd17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40dd1e:	00 00 
  40dd20:	80 7c 24 20 00       	cmpb   $0x0,0x20(%rsp)
  40dd25:	0f 84 15 05 00 00    	je     40e240 <__sprintf_chk@plt+0xb9b0>
  40dd2b:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40dd30:	0f 85 0a ff ff ff    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40dd36:	4c 39 f3             	cmp    %r14,%rbx
  40dd39:	73 09                	jae    40dd44 <__sprintf_chk@plt+0xb4b4>
  40dd3b:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40dd40:	c6 04 18 5c          	movb   $0x5c,(%rax,%rbx,1)
  40dd44:	48 8d 55 01          	lea    0x1(%rbp),%rdx
  40dd48:	48 8d 43 01          	lea    0x1(%rbx),%rax
  40dd4c:	49 39 d7             	cmp    %rdx,%r15
  40dd4f:	76 2f                	jbe    40dd80 <__sprintf_chk@plt+0xb4f0>
  40dd51:	41 0f b6 74 28 01    	movzbl 0x1(%r8,%rbp,1),%esi
  40dd57:	8d 56 d0             	lea    -0x30(%rsi),%edx
  40dd5a:	80 fa 09             	cmp    $0x9,%dl
  40dd5d:	77 21                	ja     40dd80 <__sprintf_chk@plt+0xb4f0>
  40dd5f:	49 39 c6             	cmp    %rax,%r14
  40dd62:	0f 87 98 05 00 00    	ja     40e300 <__sprintf_chk@plt+0xba70>
  40dd68:	48 8d 43 02          	lea    0x2(%rbx),%rax
  40dd6c:	49 39 c6             	cmp    %rax,%r14
  40dd6f:	76 0a                	jbe    40dd7b <__sprintf_chk@plt+0xb4eb>
  40dd71:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40dd76:	c6 44 18 02 30       	movb   $0x30,0x2(%rax,%rbx,1)
  40dd7b:	48 8d 43 03          	lea    0x3(%rbx),%rax
  40dd7f:	90                   	nop
  40dd80:	48 89 c3             	mov    %rax,%rbx
  40dd83:	41 bc 30 00 00 00    	mov    $0x30,%r12d
  40dd89:	e9 7b fd ff ff       	jmpq   40db09 <__sprintf_chk@plt+0xb279>
  40dd8e:	66 90                	xchg   %ax,%ax
  40dd90:	b8 61 00 00 00       	mov    $0x61,%eax
  40dd95:	e9 26 ff ff ff       	jmpq   40dcc0 <__sprintf_chk@plt+0xb430>
  40dd9a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40dda0:	80 7c 24 20 00       	cmpb   $0x0,0x20(%rsp)
  40dda5:	74 0f                	je     40ddb6 <__sprintf_chk@plt+0xb526>
  40dda7:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40ddac:	74 08                	je     40ddb6 <__sprintf_chk@plt+0xb526>
  40ddae:	84 c9                	test   %cl,%cl
  40ddb0:	0f 85 9a fd ff ff    	jne    40db50 <__sprintf_chk@plt+0xb2c0>
  40ddb6:	44 89 e0             	mov    %r12d,%eax
  40ddb9:	e9 f7 fe ff ff       	jmpq   40dcb5 <__sprintf_chk@plt+0xb425>
  40ddbe:	66 90                	xchg   %ax,%ax
  40ddc0:	83 7c 24 34 02       	cmpl   $0x2,0x34(%rsp)
  40ddc5:	0f 85 2d fd ff ff    	jne    40daf8 <__sprintf_chk@plt+0xb268>
  40ddcb:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40ddd0:	0f 85 6a fe ff ff    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40ddd6:	4c 39 f3             	cmp    %r14,%rbx
  40ddd9:	73 09                	jae    40dde4 <__sprintf_chk@plt+0xb554>
  40dddb:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40dde0:	c6 04 18 27          	movb   $0x27,(%rax,%rbx,1)
  40dde4:	48 8d 43 01          	lea    0x1(%rbx),%rax
  40dde8:	49 39 c6             	cmp    %rax,%r14
  40ddeb:	76 0a                	jbe    40ddf7 <__sprintf_chk@plt+0xb567>
  40dded:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40ddf2:	c6 44 18 01 5c       	movb   $0x5c,0x1(%rax,%rbx,1)
  40ddf7:	48 8d 43 02          	lea    0x2(%rbx),%rax
  40ddfb:	49 39 c6             	cmp    %rax,%r14
  40ddfe:	76 0a                	jbe    40de0a <__sprintf_chk@plt+0xb57a>
  40de00:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40de05:	c6 44 18 02 27       	movb   $0x27,0x2(%rax,%rbx,1)
  40de0a:	48 83 c3 03          	add    $0x3,%rbx
  40de0e:	e9 e5 fc ff ff       	jmpq   40daf8 <__sprintf_chk@plt+0xb268>
  40de13:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40de18:	48 83 7c 24 78 01    	cmpq   $0x1,0x78(%rsp)
  40de1e:	0f 85 9c 02 00 00    	jne    40e0c0 <__sprintf_chk@plt+0xb830>
  40de24:	4c 89 44 24 50       	mov    %r8,0x50(%rsp)
  40de29:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  40de2e:	44 89 5c 24 40       	mov    %r11d,0x40(%rsp)
  40de33:	e8 48 4a ff ff       	callq  402880 <__ctype_b_loc@plt>
  40de38:	48 8b 00             	mov    (%rax),%rax
  40de3b:	41 0f b6 d4          	movzbl %r12b,%edx
  40de3f:	44 8b 5c 24 40       	mov    0x40(%rsp),%r11d
  40de44:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
  40de49:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
  40de4e:	0f b7 14 50          	movzwl (%rax,%rdx,2),%edx
  40de52:	b8 01 00 00 00       	mov    $0x1,%eax
  40de57:	66 c1 ea 0e          	shr    $0xe,%dx
  40de5b:	83 f2 01             	xor    $0x1,%edx
  40de5e:	83 e2 01             	and    $0x1,%edx
  40de61:	22 54 24 20          	and    0x20(%rsp),%dl
  40de65:	0f 84 8d fc ff ff    	je     40daf8 <__sprintf_chk@plt+0xb268>
  40de6b:	48 01 e8             	add    %rbp,%rax
  40de6e:	0f b6 7c 24 33       	movzbl 0x33(%rsp),%edi
  40de73:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40de78:	eb 76                	jmp    40def0 <__sprintf_chk@plt+0xb660>
  40de7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40de80:	40 84 ff             	test   %dil,%dil
  40de83:	0f 85 b7 fd ff ff    	jne    40dc40 <__sprintf_chk@plt+0xb3b0>
  40de89:	4c 39 f3             	cmp    %r14,%rbx
  40de8c:	73 04                	jae    40de92 <__sprintf_chk@plt+0xb602>
  40de8e:	c6 04 19 5c          	movb   $0x5c,(%rcx,%rbx,1)
  40de92:	48 8d 73 01          	lea    0x1(%rbx),%rsi
  40de96:	49 39 f6             	cmp    %rsi,%r14
  40de99:	76 0f                	jbe    40deaa <__sprintf_chk@plt+0xb61a>
  40de9b:	44 89 e6             	mov    %r12d,%esi
  40de9e:	40 c0 ee 06          	shr    $0x6,%sil
  40dea2:	83 c6 30             	add    $0x30,%esi
  40dea5:	40 88 74 19 01       	mov    %sil,0x1(%rcx,%rbx,1)
  40deaa:	48 8d 73 02          	lea    0x2(%rbx),%rsi
  40deae:	49 39 f6             	cmp    %rsi,%r14
  40deb1:	76 12                	jbe    40dec5 <__sprintf_chk@plt+0xb635>
  40deb3:	44 89 e6             	mov    %r12d,%esi
  40deb6:	40 c0 ee 03          	shr    $0x3,%sil
  40deba:	83 e6 07             	and    $0x7,%esi
  40debd:	83 c6 30             	add    $0x30,%esi
  40dec0:	40 88 74 19 02       	mov    %sil,0x2(%rcx,%rbx,1)
  40dec5:	41 83 e4 07          	and    $0x7,%r12d
  40dec9:	48 83 c3 03          	add    $0x3,%rbx
  40decd:	41 83 c4 30          	add    $0x30,%r12d
  40ded1:	48 83 c5 01          	add    $0x1,%rbp
  40ded5:	48 39 e8             	cmp    %rbp,%rax
  40ded8:	0f 86 76 fc ff ff    	jbe    40db54 <__sprintf_chk@plt+0xb2c4>
  40dede:	4c 39 f3             	cmp    %r14,%rbx
  40dee1:	73 04                	jae    40dee7 <__sprintf_chk@plt+0xb657>
  40dee3:	44 88 24 19          	mov    %r12b,(%rcx,%rbx,1)
  40dee7:	45 0f b6 24 28       	movzbl (%r8,%rbp,1),%r12d
  40deec:	48 83 c3 01          	add    $0x1,%rbx
  40def0:	84 d2                	test   %dl,%dl
  40def2:	75 8c                	jne    40de80 <__sprintf_chk@plt+0xb5f0>
  40def4:	45 84 db             	test   %r11b,%r11b
  40def7:	74 d8                	je     40ded1 <__sprintf_chk@plt+0xb641>
  40def9:	4c 39 f3             	cmp    %r14,%rbx
  40defc:	73 04                	jae    40df02 <__sprintf_chk@plt+0xb672>
  40defe:	c6 04 19 5c          	movb   $0x5c,(%rcx,%rbx,1)
  40df02:	48 83 c3 01          	add    $0x1,%rbx
  40df06:	45 31 db             	xor    %r11d,%r11d
  40df09:	eb c6                	jmp    40ded1 <__sprintf_chk@plt+0xb641>
  40df0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40df10:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40df15:	0f 85 ed 04 00 00    	jne    40e408 <__sprintf_chk@plt+0xbb78>
  40df1b:	4d 85 db             	test   %r11,%r11
  40df1e:	0f 84 ad 03 00 00    	je     40e2d1 <__sprintf_chk@plt+0xba41>
  40df24:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40df29:	c6 44 24 20 00       	movb   $0x0,0x20(%rsp)
  40df2e:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40df34:	48 c7 44 24 60 ea 6d 	movq   $0x416dea,0x60(%rsp)
  40df3b:	41 00 
  40df3d:	bb 01 00 00 00       	mov    $0x1,%ebx
  40df42:	c6 00 27             	movb   $0x27,(%rax)
  40df45:	e9 16 fa ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40df4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40df50:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40df55:	0f 85 cd 04 00 00    	jne    40e428 <__sprintf_chk@plt+0xbb98>
  40df5b:	4d 85 db             	test   %r11,%r11
  40df5e:	0f 84 4f 03 00 00    	je     40e2b3 <__sprintf_chk@plt+0xba23>
  40df64:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  40df69:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40df6e:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40df74:	48 c7 44 24 60 eb 5f 	movq   $0x415feb,0x60(%rsp)
  40df7b:	41 00 
  40df7d:	bb 01 00 00 00       	mov    $0x1,%ebx
  40df82:	c6 00 22             	movb   $0x22,(%rax)
  40df85:	e9 d6 f9 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40df8a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40df90:	c6 44 24 33 00       	movb   $0x0,0x33(%rsp)
  40df95:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40df9a:	45 31 f6             	xor    %r14d,%r14d
  40df9d:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  40dfa4:	00 00 
  40dfa6:	31 db                	xor    %ebx,%ebx
  40dfa8:	e9 b3 f9 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40dfad:	0f 1f 00             	nopl   (%rax)
  40dfb0:	c6 44 24 33 01       	movb   $0x1,0x33(%rsp)
  40dfb5:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40dfba:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40dfc0:	48 c7 44 24 60 eb 5f 	movq   $0x415feb,0x60(%rsp)
  40dfc7:	41 00 
  40dfc9:	31 db                	xor    %ebx,%ebx
  40dfcb:	c7 44 24 34 03 00 00 	movl   $0x3,0x34(%rsp)
  40dfd2:	00 
  40dfd3:	e9 88 f9 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40dfd8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40dfdf:	00 
  40dfe0:	74 30                	je     40e012 <__sprintf_chk@plt+0xb782>
  40dfe2:	8b 5c 24 34          	mov    0x34(%rsp),%ebx
  40dfe6:	bf f5 5f 41 00       	mov    $0x415ff5,%edi
  40dfeb:	4c 89 5c 24 20       	mov    %r11,0x20(%rsp)
  40dff0:	89 de                	mov    %ebx,%esi
  40dff2:	e8 b9 f7 ff ff       	callq  40d7b0 <__sprintf_chk@plt+0xaf20>
  40dff7:	89 de                	mov    %ebx,%esi
  40dff9:	bf ea 6d 41 00       	mov    $0x416dea,%edi
  40dffe:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  40e003:	e8 a8 f7 ff ff       	callq  40d7b0 <__sprintf_chk@plt+0xaf20>
  40e008:	4c 8b 5c 24 20       	mov    0x20(%rsp),%r11
  40e00d:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  40e012:	31 db                	xor    %ebx,%ebx
  40e014:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40e019:	75 29                	jne    40e044 <__sprintf_chk@plt+0xb7b4>
  40e01b:	48 8b 54 24 70       	mov    0x70(%rsp),%rdx
  40e020:	0f b6 02             	movzbl (%rdx),%eax
  40e023:	84 c0                	test   %al,%al
  40e025:	74 1d                	je     40e044 <__sprintf_chk@plt+0xb7b4>
  40e027:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40e02c:	0f 1f 40 00          	nopl   0x0(%rax)
  40e030:	4c 39 db             	cmp    %r11,%rbx
  40e033:	73 03                	jae    40e038 <__sprintf_chk@plt+0xb7a8>
  40e035:	88 04 19             	mov    %al,(%rcx,%rbx,1)
  40e038:	48 83 c3 01          	add    $0x1,%rbx
  40e03c:	0f b6 04 1a          	movzbl (%rdx,%rbx,1),%eax
  40e040:	84 c0                	test   %al,%al
  40e042:	75 ec                	jne    40e030 <__sprintf_chk@plt+0xb7a0>
  40e044:	48 8b 6c 24 68       	mov    0x68(%rsp),%rbp
  40e049:	4c 89 5c 24 38       	mov    %r11,0x38(%rsp)
  40e04e:	48 89 ef             	mov    %rbp,%rdi
  40e051:	e8 2a 43 ff ff       	callq  402380 <strlen@plt>
  40e056:	48 89 6c 24 60       	mov    %rbp,0x60(%rsp)
  40e05b:	49 89 c6             	mov    %rax,%r14
  40e05e:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40e063:	4c 8b 5c 24 38       	mov    0x38(%rsp),%r11
  40e068:	e9 f3 f8 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e06d:	0f 1f 00             	nopl   (%rax)
  40e070:	c6 44 24 33 01       	movb   $0x1,0x33(%rsp)
  40e075:	c6 44 24 20 00       	movb   $0x0,0x20(%rsp)
  40e07a:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40e080:	48 c7 44 24 60 ea 6d 	movq   $0x416dea,0x60(%rsp)
  40e087:	41 00 
  40e089:	31 db                	xor    %ebx,%ebx
  40e08b:	c7 44 24 34 02 00 00 	movl   $0x2,0x34(%rsp)
  40e092:	00 
  40e093:	e9 c8 f8 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e098:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40e09f:	00 
  40e0a0:	4d 8d 2c 28          	lea    (%r8,%rbp,1),%r13
  40e0a4:	45 31 db             	xor    %r11d,%r11d
  40e0a7:	e9 74 f9 ff ff       	jmpq   40da20 <__sprintf_chk@plt+0xb190>
  40e0ac:	0f 1f 40 00          	nopl   0x0(%rax)
  40e0b0:	45 31 db             	xor    %r11d,%r11d
  40e0b3:	e9 68 f9 ff ff       	jmpq   40da20 <__sprintf_chk@plt+0xb190>
  40e0b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40e0bf:	00 
  40e0c0:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
  40e0c4:	48 c7 84 24 b0 00 00 	movq   $0x0,0xb0(%rsp)
  40e0cb:	00 00 00 00 00 
  40e0d0:	0f 84 af 01 00 00    	je     40e285 <__sprintf_chk@plt+0xb9f5>
  40e0d6:	be 01 00 00 00       	mov    $0x1,%esi
  40e0db:	31 c0                	xor    %eax,%eax
  40e0dd:	48 89 9c 24 80 00 00 	mov    %rbx,0x80(%rsp)
  40e0e4:	00 
  40e0e5:	44 88 a4 24 96 00 00 	mov    %r12b,0x96(%rsp)
  40e0ec:	00 
  40e0ed:	4c 89 ac 24 98 00 00 	mov    %r13,0x98(%rsp)
  40e0f4:	00 
  40e0f5:	48 89 c3             	mov    %rax,%rbx
  40e0f8:	48 89 6c 24 48       	mov    %rbp,0x48(%rsp)
  40e0fd:	4c 89 8c 24 88 00 00 	mov    %r9,0x88(%rsp)
  40e104:	00 
  40e105:	41 89 f4             	mov    %esi,%r12d
  40e108:	44 88 9c 24 97 00 00 	mov    %r11b,0x97(%rsp)
  40e10f:	00 
  40e110:	4c 89 74 24 50       	mov    %r14,0x50(%rsp)
  40e115:	4d 89 c5             	mov    %r8,%r13
  40e118:	4c 89 7c 24 40       	mov    %r15,0x40(%rsp)
  40e11d:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
  40e122:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
  40e127:	48 8d 8c 24 b0 00 00 	lea    0xb0(%rsp),%rcx
  40e12e:	00 
  40e12f:	48 8d bc 24 ac 00 00 	lea    0xac(%rsp),%rdi
  40e136:	00 
  40e137:	4c 8d 34 03          	lea    (%rbx,%rax,1),%r14
  40e13b:	4f 8d 7c 35 00       	lea    0x0(%r13,%r14,1),%r15
  40e140:	4c 29 f2             	sub    %r14,%rdx
  40e143:	4c 89 fe             	mov    %r15,%rsi
  40e146:	e8 75 42 ff ff       	callq  4023c0 <mbrtowc@plt>
  40e14b:	48 85 c0             	test   %rax,%rax
  40e14e:	48 89 c5             	mov    %rax,%rbp
  40e151:	0f 84 b7 01 00 00    	je     40e30e <__sprintf_chk@plt+0xba7e>
  40e157:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  40e15b:	0f 84 ef 01 00 00    	je     40e350 <__sprintf_chk@plt+0xbac0>
  40e161:	48 83 f8 fe          	cmp    $0xfffffffffffffffe,%rax
  40e165:	0f 84 26 02 00 00    	je     40e391 <__sprintf_chk@plt+0xbb01>
  40e16b:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40e170:	74 4d                	je     40e1bf <__sprintf_chk@plt+0xb92f>
  40e172:	83 7c 24 34 02       	cmpl   $0x2,0x34(%rsp)
  40e177:	75 46                	jne    40e1bf <__sprintf_chk@plt+0xb92f>
  40e179:	48 83 f8 01          	cmp    $0x1,%rax
  40e17d:	74 40                	je     40e1bf <__sprintf_chk@plt+0xb92f>
  40e17f:	ba 01 00 00 00       	mov    $0x1,%edx
  40e184:	b8 01 00 00 00       	mov    $0x1,%eax
  40e189:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40e190:	41 0f b6 3c 17       	movzbl (%r15,%rdx,1),%edi
  40e195:	8d 4f a5             	lea    -0x5b(%rdi),%ecx
  40e198:	80 f9 21             	cmp    $0x21,%cl
  40e19b:	77 19                	ja     40e1b6 <__sprintf_chk@plt+0xb926>
  40e19d:	48 89 c6             	mov    %rax,%rsi
  40e1a0:	48 bf 2b 00 00 00 02 	movabs $0x20000002b,%rdi
  40e1a7:	00 00 00 
  40e1aa:	48 d3 e6             	shl    %cl,%rsi
  40e1ad:	48 85 fe             	test   %rdi,%rsi
  40e1b0:	0f 85 aa 00 00 00    	jne    40e260 <__sprintf_chk@plt+0xb9d0>
  40e1b6:	48 83 c2 01          	add    $0x1,%rdx
  40e1ba:	48 39 ea             	cmp    %rbp,%rdx
  40e1bd:	75 d1                	jne    40e190 <__sprintf_chk@plt+0xb900>
  40e1bf:	8b bc 24 ac 00 00 00 	mov    0xac(%rsp),%edi
  40e1c6:	e8 75 46 ff ff       	callq  402840 <iswprint@plt>
  40e1cb:	48 8d bc 24 b0 00 00 	lea    0xb0(%rsp),%rdi
  40e1d2:	00 
  40e1d3:	85 c0                	test   %eax,%eax
  40e1d5:	b8 00 00 00 00       	mov    $0x0,%eax
  40e1da:	44 0f 44 e0          	cmove  %eax,%r12d
  40e1de:	48 01 eb             	add    %rbp,%rbx
  40e1e1:	e8 4a 46 ff ff       	callq  402830 <mbsinit@plt>
  40e1e6:	85 c0                	test   %eax,%eax
  40e1e8:	0f 84 2f ff ff ff    	je     40e11d <__sprintf_chk@plt+0xb88d>
  40e1ee:	44 89 e6             	mov    %r12d,%esi
  40e1f1:	44 0f b6 9c 24 97 00 	movzbl 0x97(%rsp),%r11d
  40e1f8:	00 00 
  40e1fa:	44 0f b6 a4 24 96 00 	movzbl 0x96(%rsp),%r12d
  40e201:	00 00 
  40e203:	48 89 d8             	mov    %rbx,%rax
  40e206:	48 8b 6c 24 48       	mov    0x48(%rsp),%rbp
  40e20b:	4c 8b 8c 24 88 00 00 	mov    0x88(%rsp),%r9
  40e212:	00 
  40e213:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
  40e21a:	00 
  40e21b:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
  40e220:	89 f2                	mov    %esi,%edx
  40e222:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  40e227:	4d 89 e8             	mov    %r13,%r8
  40e22a:	83 f2 01             	xor    $0x1,%edx
  40e22d:	48 83 f8 01          	cmp    $0x1,%rax
  40e231:	0f 86 2a fc ff ff    	jbe    40de61 <__sprintf_chk@plt+0xb5d1>
  40e237:	22 54 24 20          	and    0x20(%rsp),%dl
  40e23b:	e9 2b fc ff ff       	jmpq   40de6b <__sprintf_chk@plt+0xb5db>
  40e240:	f6 84 24 90 00 00 00 	testb  $0x1,0x90(%rsp)
  40e247:	01 
  40e248:	0f 84 aa f8 ff ff    	je     40daf8 <__sprintf_chk@plt+0xb268>
  40e24e:	48 83 c5 01          	add    $0x1,%rbp
  40e252:	e9 2f f7 ff ff       	jmpq   40d986 <__sprintf_chk@plt+0xb0f6>
  40e257:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40e25e:	00 00 
  40e260:	4c 8b 5c 24 50       	mov    0x50(%rsp),%r11
  40e265:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  40e26a:	e9 d7 f9 ff ff       	jmpq   40dc46 <__sprintf_chk@plt+0xb3b6>
  40e26f:	90                   	nop
  40e270:	41 80 78 01 00       	cmpb   $0x0,0x1(%r8)
  40e275:	0f 95 c0             	setne  %al
  40e278:	e9 94 f9 ff ff       	jmpq   40dc11 <__sprintf_chk@plt+0xb381>
  40e27d:	0f 1f 00             	nopl   (%rax)
  40e280:	e8 9b 3f ff ff       	callq  402220 <abort@plt>
  40e285:	4c 89 c7             	mov    %r8,%rdi
  40e288:	4c 89 4c 24 50       	mov    %r9,0x50(%rsp)
  40e28d:	44 89 5c 24 48       	mov    %r11d,0x48(%rsp)
  40e292:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  40e297:	e8 e4 40 ff ff       	callq  402380 <strlen@plt>
  40e29c:	4c 8b 4c 24 50       	mov    0x50(%rsp),%r9
  40e2a1:	49 89 c7             	mov    %rax,%r15
  40e2a4:	44 8b 5c 24 48       	mov    0x48(%rsp),%r11d
  40e2a9:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
  40e2ae:	e9 23 fe ff ff       	jmpq   40e0d6 <__sprintf_chk@plt+0xb846>
  40e2b3:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40e2b8:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40e2be:	48 c7 44 24 60 eb 5f 	movq   $0x415feb,0x60(%rsp)
  40e2c5:	41 00 
  40e2c7:	bb 01 00 00 00       	mov    $0x1,%ebx
  40e2cc:	e9 8f f6 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e2d1:	c6 44 24 20 00       	movb   $0x0,0x20(%rsp)
  40e2d6:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40e2dc:	48 c7 44 24 60 ea 6d 	movq   $0x416dea,0x60(%rsp)
  40e2e3:	41 00 
  40e2e5:	bb 01 00 00 00       	mov    $0x1,%ebx
  40e2ea:	e9 71 f6 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e2ef:	80 7c 24 33 00       	cmpb   $0x0,0x33(%rsp)
  40e2f4:	0f 84 c6 f9 ff ff    	je     40dcc0 <__sprintf_chk@plt+0xb430>
  40e2fa:	e9 41 f9 ff ff       	jmpq   40dc40 <__sprintf_chk@plt+0xb3b0>
  40e2ff:	90                   	nop
  40e300:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  40e305:	c6 04 06 30          	movb   $0x30,(%rsi,%rax,1)
  40e309:	e9 5a fa ff ff       	jmpq   40dd68 <__sprintf_chk@plt+0xb4d8>
  40e30e:	44 89 e2             	mov    %r12d,%edx
  40e311:	48 89 d8             	mov    %rbx,%rax
  40e314:	48 8b 6c 24 48       	mov    0x48(%rsp),%rbp
  40e319:	4c 8b 8c 24 88 00 00 	mov    0x88(%rsp),%r9
  40e320:	00 
  40e321:	44 0f b6 9c 24 97 00 	movzbl 0x97(%rsp),%r11d
  40e328:	00 00 
  40e32a:	4d 89 e8             	mov    %r13,%r8
  40e32d:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
  40e334:	00 
  40e335:	44 0f b6 a4 24 96 00 	movzbl 0x96(%rsp),%r12d
  40e33c:	00 00 
  40e33e:	83 f2 01             	xor    $0x1,%edx
  40e341:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
  40e346:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  40e34b:	e9 dd fe ff ff       	jmpq   40e22d <__sprintf_chk@plt+0xb99d>
  40e350:	48 89 d8             	mov    %rbx,%rax
  40e353:	48 8b 6c 24 48       	mov    0x48(%rsp),%rbp
  40e358:	4c 8b 8c 24 88 00 00 	mov    0x88(%rsp),%r9
  40e35f:	00 
  40e360:	44 0f b6 a4 24 96 00 	movzbl 0x96(%rsp),%r12d
  40e367:	00 00 
  40e369:	44 0f b6 9c 24 97 00 	movzbl 0x97(%rsp),%r11d
  40e370:	00 00 
  40e372:	4d 89 e8             	mov    %r13,%r8
  40e375:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
  40e37c:	00 
  40e37d:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
  40e382:	ba 01 00 00 00       	mov    $0x1,%edx
  40e387:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  40e38c:	e9 9c fe ff ff       	jmpq   40e22d <__sprintf_chk@plt+0xb99d>
  40e391:	4d 89 fa             	mov    %r15,%r10
  40e394:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
  40e399:	4c 89 f6             	mov    %r14,%rsi
  40e39c:	48 89 d8             	mov    %rbx,%rax
  40e39f:	4d 89 e8             	mov    %r13,%r8
  40e3a2:	48 8b 6c 24 48       	mov    0x48(%rsp),%rbp
  40e3a7:	4c 8b 8c 24 88 00 00 	mov    0x88(%rsp),%r9
  40e3ae:	00 
  40e3af:	44 0f b6 a4 24 96 00 	movzbl 0x96(%rsp),%r12d
  40e3b6:	00 00 
  40e3b8:	49 39 f7             	cmp    %rsi,%r15
  40e3bb:	44 0f b6 9c 24 97 00 	movzbl 0x97(%rsp),%r11d
  40e3c2:	00 00 
  40e3c4:	48 8b 9c 24 80 00 00 	mov    0x80(%rsp),%rbx
  40e3cb:	00 
  40e3cc:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
  40e3d1:	4c 8b ac 24 98 00 00 	mov    0x98(%rsp),%r13
  40e3d8:	00 
  40e3d9:	76 23                	jbe    40e3fe <__sprintf_chk@plt+0xbb6e>
  40e3db:	41 80 3a 00          	cmpb   $0x0,(%r10)
  40e3df:	75 0f                	jne    40e3f0 <__sprintf_chk@plt+0xbb60>
  40e3e1:	eb 1b                	jmp    40e3fe <__sprintf_chk@plt+0xbb6e>
  40e3e3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40e3e8:	41 80 7c 05 00 00    	cmpb   $0x0,0x0(%r13,%rax,1)
  40e3ee:	74 0e                	je     40e3fe <__sprintf_chk@plt+0xbb6e>
  40e3f0:	48 83 c0 01          	add    $0x1,%rax
  40e3f4:	48 8d 54 05 00       	lea    0x0(%rbp,%rax,1),%rdx
  40e3f9:	49 39 d7             	cmp    %rdx,%r15
  40e3fc:	77 ea                	ja     40e3e8 <__sprintf_chk@plt+0xbb58>
  40e3fe:	ba 01 00 00 00       	mov    $0x1,%edx
  40e403:	e9 25 fe ff ff       	jmpq   40e22d <__sprintf_chk@plt+0xb99d>
  40e408:	c6 44 24 20 00       	movb   $0x0,0x20(%rsp)
  40e40d:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40e413:	48 c7 44 24 60 ea 6d 	movq   $0x416dea,0x60(%rsp)
  40e41a:	41 00 
  40e41c:	31 db                	xor    %ebx,%ebx
  40e41e:	e9 3d f5 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e423:	e8 78 3f ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  40e428:	c6 44 24 20 01       	movb   $0x1,0x20(%rsp)
  40e42d:	41 be 01 00 00 00    	mov    $0x1,%r14d
  40e433:	48 c7 44 24 60 eb 5f 	movq   $0x415feb,0x60(%rsp)
  40e43a:	41 00 
  40e43c:	31 db                	xor    %ebx,%ebx
  40e43e:	e9 1d f5 ff ff       	jmpq   40d960 <__sprintf_chk@plt+0xb0d0>
  40e443:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e44a:	84 00 00 00 00 00 
  40e450:	41 57                	push   %r15
  40e452:	4c 63 ff             	movslq %edi,%r15
  40e455:	41 56                	push   %r14
  40e457:	41 55                	push   %r13
  40e459:	41 54                	push   %r12
  40e45b:	55                   	push   %rbp
  40e45c:	53                   	push   %rbx
  40e45d:	48 89 cb             	mov    %rcx,%rbx
  40e460:	48 83 ec 48          	sub    $0x48,%rsp
  40e464:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  40e469:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
  40e46e:	e8 bd 3d ff ff       	callq  402230 <__errno_location@plt>
  40e473:	49 89 c5             	mov    %rax,%r13
  40e476:	8b 00                	mov    (%rax),%eax
  40e478:	45 85 ff             	test   %r15d,%r15d
  40e47b:	4c 8b 25 56 c1 20 00 	mov    0x20c156(%rip),%r12        # 61a5d8 <_fini@@Base+0x2086dc>
  40e482:	89 44 24 34          	mov    %eax,0x34(%rsp)
  40e486:	0f 88 6b 01 00 00    	js     40e5f7 <__sprintf_chk@plt+0xbd67>
  40e48c:	44 3b 3d 5d c1 20 00 	cmp    0x20c15d(%rip),%r15d        # 61a5f0 <_fini@@Base+0x2086f4>
  40e493:	72 65                	jb     40e4fa <__sprintf_chk@plt+0xbc6a>
  40e495:	41 8d 6f 01          	lea    0x1(%r15),%ebp
  40e499:	41 89 ee             	mov    %ebp,%r14d
  40e49c:	4c 89 f6             	mov    %r14,%rsi
  40e49f:	48 c1 e6 04          	shl    $0x4,%rsi
  40e4a3:	49 81 fc e0 a5 61 00 	cmp    $0x61a5e0,%r12
  40e4aa:	0f 85 30 01 00 00    	jne    40e5e0 <__sprintf_chk@plt+0xbd50>
  40e4b0:	31 ff                	xor    %edi,%edi
  40e4b2:	e8 d9 27 00 00       	callq  410c90 <__sprintf_chk@plt+0xe400>
  40e4b7:	48 8b 35 22 c1 20 00 	mov    0x20c122(%rip),%rsi        # 61a5e0 <_fini@@Base+0x2086e4>
  40e4be:	48 8b 3d 23 c1 20 00 	mov    0x20c123(%rip),%rdi        # 61a5e8 <_fini@@Base+0x2086ec>
  40e4c5:	49 89 c4             	mov    %rax,%r12
  40e4c8:	48 89 05 09 c1 20 00 	mov    %rax,0x20c109(%rip)        # 61a5d8 <_fini@@Base+0x2086dc>
  40e4cf:	48 89 30             	mov    %rsi,(%rax)
  40e4d2:	48 89 78 08          	mov    %rdi,0x8(%rax)
  40e4d6:	8b 3d 14 c1 20 00    	mov    0x20c114(%rip),%edi        # 61a5f0 <_fini@@Base+0x2086f4>
  40e4dc:	4c 89 f2             	mov    %r14,%rdx
  40e4df:	31 f6                	xor    %esi,%esi
  40e4e1:	48 29 fa             	sub    %rdi,%rdx
  40e4e4:	48 c1 e7 04          	shl    $0x4,%rdi
  40e4e8:	48 c1 e2 04          	shl    $0x4,%rdx
  40e4ec:	4c 01 e7             	add    %r12,%rdi
  40e4ef:	e8 8c 3f ff ff       	callq  402480 <memset@plt>
  40e4f4:	89 2d f6 c0 20 00    	mov    %ebp,0x20c0f6(%rip)        # 61a5f0 <_fini@@Base+0x2086f4>
  40e4fa:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40e4fe:	49 c1 e7 04          	shl    $0x4,%r15
  40e502:	8b 6b 04             	mov    0x4(%rbx),%ebp
  40e505:	4d 01 fc             	add    %r15,%r12
  40e508:	44 8b 03             	mov    (%rbx),%r8d
  40e50b:	4c 8d 7b 08          	lea    0x8(%rbx),%r15
  40e50f:	4d 8b 1c 24          	mov    (%r12),%r11
  40e513:	4d 8b 74 24 08       	mov    0x8(%r12),%r14
  40e518:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40e51d:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40e521:	83 cd 01             	or     $0x1,%ebp
  40e524:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40e529:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  40e52e:	41 89 e9             	mov    %ebp,%r9d
  40e531:	4c 89 de             	mov    %r11,%rsi
  40e534:	4c 89 3c 24          	mov    %r15,(%rsp)
  40e538:	4c 89 f7             	mov    %r14,%rdi
  40e53b:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40e540:	4c 89 5c 24 38       	mov    %r11,0x38(%rsp)
  40e545:	e8 56 f3 ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40e54a:	4c 8b 5c 24 38       	mov    0x38(%rsp),%r11
  40e54f:	49 39 c3             	cmp    %rax,%r11
  40e552:	77 6b                	ja     40e5bf <__sprintf_chk@plt+0xbd2f>
  40e554:	48 8d 70 01          	lea    0x1(%rax),%rsi
  40e558:	49 81 fe 20 b2 61 00 	cmp    $0x61b220,%r14
  40e55f:	49 89 34 24          	mov    %rsi,(%r12)
  40e563:	74 12                	je     40e577 <__sprintf_chk@plt+0xbce7>
  40e565:	4c 89 f7             	mov    %r14,%rdi
  40e568:	48 89 74 24 38       	mov    %rsi,0x38(%rsp)
  40e56d:	e8 7e 3c ff ff       	callq  4021f0 <free@plt>
  40e572:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
  40e577:	48 89 f7             	mov    %rsi,%rdi
  40e57a:	48 89 74 24 38       	mov    %rsi,0x38(%rsp)
  40e57f:	e8 bc 26 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40e584:	49 89 44 24 08       	mov    %rax,0x8(%r12)
  40e589:	49 89 c6             	mov    %rax,%r14
  40e58c:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40e590:	44 8b 03             	mov    (%rbx),%r8d
  40e593:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40e598:	41 89 e9             	mov    %ebp,%r9d
  40e59b:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  40e5a0:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
  40e5a5:	4c 89 f7             	mov    %r14,%rdi
  40e5a8:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40e5ad:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40e5b1:	4c 89 3c 24          	mov    %r15,(%rsp)
  40e5b5:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40e5ba:	e8 e1 f2 ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40e5bf:	8b 44 24 34          	mov    0x34(%rsp),%eax
  40e5c3:	41 89 45 00          	mov    %eax,0x0(%r13)
  40e5c7:	48 83 c4 48          	add    $0x48,%rsp
  40e5cb:	4c 89 f0             	mov    %r14,%rax
  40e5ce:	5b                   	pop    %rbx
  40e5cf:	5d                   	pop    %rbp
  40e5d0:	41 5c                	pop    %r12
  40e5d2:	41 5d                	pop    %r13
  40e5d4:	41 5e                	pop    %r14
  40e5d6:	41 5f                	pop    %r15
  40e5d8:	c3                   	retq   
  40e5d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40e5e0:	4c 89 e7             	mov    %r12,%rdi
  40e5e3:	e8 a8 26 00 00       	callq  410c90 <__sprintf_chk@plt+0xe400>
  40e5e8:	49 89 c4             	mov    %rax,%r12
  40e5eb:	48 89 05 e6 bf 20 00 	mov    %rax,0x20bfe6(%rip)        # 61a5d8 <_fini@@Base+0x2086dc>
  40e5f2:	e9 df fe ff ff       	jmpq   40e4d6 <__sprintf_chk@plt+0xbc46>
  40e5f7:	e8 24 3c ff ff       	callq  402220 <abort@plt>
  40e5fc:	0f 1f 40 00          	nopl   0x0(%rax)
  40e600:	41 54                	push   %r12
  40e602:	55                   	push   %rbp
  40e603:	48 89 fd             	mov    %rdi,%rbp
  40e606:	53                   	push   %rbx
  40e607:	e8 24 3c ff ff       	callq  402230 <__errno_location@plt>
  40e60c:	44 8b 20             	mov    (%rax),%r12d
  40e60f:	48 85 ed             	test   %rbp,%rbp
  40e612:	bf 20 b3 61 00       	mov    $0x61b320,%edi
  40e617:	48 89 c3             	mov    %rax,%rbx
  40e61a:	48 0f 45 fd          	cmovne %rbp,%rdi
  40e61e:	be 38 00 00 00       	mov    $0x38,%esi
  40e623:	e8 d8 27 00 00       	callq  410e00 <__sprintf_chk@plt+0xe570>
  40e628:	44 89 23             	mov    %r12d,(%rbx)
  40e62b:	5b                   	pop    %rbx
  40e62c:	5d                   	pop    %rbp
  40e62d:	41 5c                	pop    %r12
  40e62f:	c3                   	retq   
  40e630:	48 85 ff             	test   %rdi,%rdi
  40e633:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e638:	48 0f 45 c7          	cmovne %rdi,%rax
  40e63c:	8b 00                	mov    (%rax),%eax
  40e63e:	c3                   	retq   
  40e63f:	90                   	nop
  40e640:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e645:	48 85 ff             	test   %rdi,%rdi
  40e648:	48 0f 45 c7          	cmovne %rdi,%rax
  40e64c:	89 30                	mov    %esi,(%rax)
  40e64e:	c3                   	retq   
  40e64f:	90                   	nop
  40e650:	48 85 ff             	test   %rdi,%rdi
  40e653:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e658:	89 f1                	mov    %esi,%ecx
  40e65a:	48 0f 45 c7          	cmovne %rdi,%rax
  40e65e:	40 c0 ee 05          	shr    $0x5,%sil
  40e662:	83 e1 1f             	and    $0x1f,%ecx
  40e665:	40 0f b6 f6          	movzbl %sil,%esi
  40e669:	48 8d 34 b0          	lea    (%rax,%rsi,4),%rsi
  40e66d:	8b 7e 08             	mov    0x8(%rsi),%edi
  40e670:	89 f8                	mov    %edi,%eax
  40e672:	d3 e8                	shr    %cl,%eax
  40e674:	31 c2                	xor    %eax,%edx
  40e676:	83 e0 01             	and    $0x1,%eax
  40e679:	83 e2 01             	and    $0x1,%edx
  40e67c:	d3 e2                	shl    %cl,%edx
  40e67e:	31 fa                	xor    %edi,%edx
  40e680:	89 56 08             	mov    %edx,0x8(%rsi)
  40e683:	c3                   	retq   
  40e684:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e68b:	00 00 00 00 00 
  40e690:	48 85 ff             	test   %rdi,%rdi
  40e693:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e698:	48 0f 44 f8          	cmove  %rax,%rdi
  40e69c:	8b 47 04             	mov    0x4(%rdi),%eax
  40e69f:	89 77 04             	mov    %esi,0x4(%rdi)
  40e6a2:	c3                   	retq   
  40e6a3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e6aa:	84 00 00 00 00 00 
  40e6b0:	48 83 ec 08          	sub    $0x8,%rsp
  40e6b4:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e6b9:	48 85 ff             	test   %rdi,%rdi
  40e6bc:	48 0f 44 f8          	cmove  %rax,%rdi
  40e6c0:	48 85 f6             	test   %rsi,%rsi
  40e6c3:	c7 07 08 00 00 00    	movl   $0x8,(%rdi)
  40e6c9:	74 12                	je     40e6dd <__sprintf_chk@plt+0xbe4d>
  40e6cb:	48 85 d2             	test   %rdx,%rdx
  40e6ce:	74 0d                	je     40e6dd <__sprintf_chk@plt+0xbe4d>
  40e6d0:	48 89 77 28          	mov    %rsi,0x28(%rdi)
  40e6d4:	48 89 57 30          	mov    %rdx,0x30(%rdi)
  40e6d8:	48 83 c4 08          	add    $0x8,%rsp
  40e6dc:	c3                   	retq   
  40e6dd:	e8 3e 3b ff ff       	callq  402220 <abort@plt>
  40e6e2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e6e9:	1f 84 00 00 00 00 00 
  40e6f0:	41 57                	push   %r15
  40e6f2:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e6f7:	49 89 cf             	mov    %rcx,%r15
  40e6fa:	41 56                	push   %r14
  40e6fc:	49 89 d6             	mov    %rdx,%r14
  40e6ff:	41 55                	push   %r13
  40e701:	49 89 f5             	mov    %rsi,%r13
  40e704:	41 54                	push   %r12
  40e706:	55                   	push   %rbp
  40e707:	53                   	push   %rbx
  40e708:	4c 89 c3             	mov    %r8,%rbx
  40e70b:	48 83 ec 28          	sub    $0x28,%rsp
  40e70f:	4d 85 c0             	test   %r8,%r8
  40e712:	48 0f 44 d8          	cmove  %rax,%rbx
  40e716:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  40e71b:	e8 10 3b ff ff       	callq  402230 <__errno_location@plt>
  40e720:	44 8b 20             	mov    (%rax),%r12d
  40e723:	48 89 c5             	mov    %rax,%rbp
  40e726:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40e72a:	44 8b 4b 04          	mov    0x4(%rbx),%r9d
  40e72e:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  40e733:	4c 89 f9             	mov    %r15,%rcx
  40e736:	4c 89 f2             	mov    %r14,%rdx
  40e739:	4c 89 ee             	mov    %r13,%rsi
  40e73c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40e741:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40e745:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40e74a:	48 8d 43 08          	lea    0x8(%rbx),%rax
  40e74e:	48 89 04 24          	mov    %rax,(%rsp)
  40e752:	44 8b 03             	mov    (%rbx),%r8d
  40e755:	e8 46 f1 ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40e75a:	44 89 65 00          	mov    %r12d,0x0(%rbp)
  40e75e:	48 83 c4 28          	add    $0x28,%rsp
  40e762:	5b                   	pop    %rbx
  40e763:	5d                   	pop    %rbp
  40e764:	41 5c                	pop    %r12
  40e766:	41 5d                	pop    %r13
  40e768:	41 5e                	pop    %r14
  40e76a:	41 5f                	pop    %r15
  40e76c:	c3                   	retq   
  40e76d:	0f 1f 00             	nopl   (%rax)
  40e770:	41 57                	push   %r15
  40e772:	b8 20 b3 61 00       	mov    $0x61b320,%eax
  40e777:	41 56                	push   %r14
  40e779:	49 89 f6             	mov    %rsi,%r14
  40e77c:	41 55                	push   %r13
  40e77e:	49 89 fd             	mov    %rdi,%r13
  40e781:	41 54                	push   %r12
  40e783:	49 89 d4             	mov    %rdx,%r12
  40e786:	55                   	push   %rbp
  40e787:	53                   	push   %rbx
  40e788:	48 89 cb             	mov    %rcx,%rbx
  40e78b:	48 83 ec 48          	sub    $0x48,%rsp
  40e78f:	48 85 c9             	test   %rcx,%rcx
  40e792:	48 0f 44 d8          	cmove  %rax,%rbx
  40e796:	31 ed                	xor    %ebp,%ebp
  40e798:	e8 93 3a ff ff       	callq  402230 <__errno_location@plt>
  40e79d:	49 89 c7             	mov    %rax,%r15
  40e7a0:	8b 00                	mov    (%rax),%eax
  40e7a2:	4d 85 e4             	test   %r12,%r12
  40e7a5:	40 0f 94 c5          	sete   %bpl
  40e7a9:	0b 6b 04             	or     0x4(%rbx),%ebp
  40e7ac:	4c 8d 53 08          	lea    0x8(%rbx),%r10
  40e7b0:	4c 89 f1             	mov    %r14,%rcx
  40e7b3:	4c 89 ea             	mov    %r13,%rdx
  40e7b6:	31 f6                	xor    %esi,%esi
  40e7b8:	89 44 24 1c          	mov    %eax,0x1c(%rsp)
  40e7bc:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40e7c0:	31 ff                	xor    %edi,%edi
  40e7c2:	4c 89 54 24 30       	mov    %r10,0x30(%rsp)
  40e7c7:	41 89 e9             	mov    %ebp,%r9d
  40e7ca:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40e7cf:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40e7d3:	4c 89 14 24          	mov    %r10,(%rsp)
  40e7d7:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40e7dc:	44 8b 03             	mov    (%rbx),%r8d
  40e7df:	e8 bc f0 ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40e7e4:	48 8d 70 01          	lea    0x1(%rax),%rsi
  40e7e8:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  40e7ed:	48 89 f7             	mov    %rsi,%rdi
  40e7f0:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
  40e7f5:	e8 46 24 00 00       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  40e7fa:	48 89 c7             	mov    %rax,%rdi
  40e7fd:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40e802:	48 8b 43 30          	mov    0x30(%rbx),%rax
  40e806:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
  40e80b:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
  40e810:	41 89 e9             	mov    %ebp,%r9d
  40e813:	4c 89 f1             	mov    %r14,%rcx
  40e816:	4c 89 ea             	mov    %r13,%rdx
  40e819:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40e81e:	48 8b 43 28          	mov    0x28(%rbx),%rax
  40e822:	4c 89 14 24          	mov    %r10,(%rsp)
  40e826:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40e82b:	44 8b 03             	mov    (%rbx),%r8d
  40e82e:	e8 6d f0 ff ff       	callq  40d8a0 <__sprintf_chk@plt+0xb010>
  40e833:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
  40e837:	4d 85 e4             	test   %r12,%r12
  40e83a:	41 89 07             	mov    %eax,(%r15)
  40e83d:	74 09                	je     40e848 <__sprintf_chk@plt+0xbfb8>
  40e83f:	4c 8b 5c 24 38       	mov    0x38(%rsp),%r11
  40e844:	4d 89 1c 24          	mov    %r11,(%r12)
  40e848:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  40e84d:	48 83 c4 48          	add    $0x48,%rsp
  40e851:	5b                   	pop    %rbx
  40e852:	5d                   	pop    %rbp
  40e853:	41 5c                	pop    %r12
  40e855:	41 5d                	pop    %r13
  40e857:	41 5e                	pop    %r14
  40e859:	41 5f                	pop    %r15
  40e85b:	c3                   	retq   
  40e85c:	0f 1f 40 00          	nopl   0x0(%rax)
  40e860:	48 89 d1             	mov    %rdx,%rcx
  40e863:	31 d2                	xor    %edx,%edx
  40e865:	e9 06 ff ff ff       	jmpq   40e770 <__sprintf_chk@plt+0xbee0>
  40e86a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40e870:	41 54                	push   %r12
  40e872:	8b 05 78 bd 20 00    	mov    0x20bd78(%rip),%eax        # 61a5f0 <_fini@@Base+0x2086f4>
  40e878:	4c 8b 25 59 bd 20 00 	mov    0x20bd59(%rip),%r12        # 61a5d8 <_fini@@Base+0x2086dc>
  40e87f:	55                   	push   %rbp
  40e880:	83 f8 01             	cmp    $0x1,%eax
  40e883:	53                   	push   %rbx
  40e884:	76 24                	jbe    40e8aa <__sprintf_chk@plt+0xc01a>
  40e886:	83 e8 02             	sub    $0x2,%eax
  40e889:	4c 89 e3             	mov    %r12,%rbx
  40e88c:	48 c1 e0 04          	shl    $0x4,%rax
  40e890:	49 8d 6c 04 10       	lea    0x10(%r12,%rax,1),%rbp
  40e895:	0f 1f 00             	nopl   (%rax)
  40e898:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
  40e89c:	48 83 c3 10          	add    $0x10,%rbx
  40e8a0:	e8 4b 39 ff ff       	callq  4021f0 <free@plt>
  40e8a5:	48 39 eb             	cmp    %rbp,%rbx
  40e8a8:	75 ee                	jne    40e898 <__sprintf_chk@plt+0xc008>
  40e8aa:	49 8b 7c 24 08       	mov    0x8(%r12),%rdi
  40e8af:	48 81 ff 20 b2 61 00 	cmp    $0x61b220,%rdi
  40e8b6:	74 1b                	je     40e8d3 <__sprintf_chk@plt+0xc043>
  40e8b8:	e8 33 39 ff ff       	callq  4021f0 <free@plt>
  40e8bd:	48 c7 05 18 bd 20 00 	movq   $0x100,0x20bd18(%rip)        # 61a5e0 <_fini@@Base+0x2086e4>
  40e8c4:	00 01 00 00 
  40e8c8:	48 c7 05 15 bd 20 00 	movq   $0x61b220,0x20bd15(%rip)        # 61a5e8 <_fini@@Base+0x2086ec>
  40e8cf:	20 b2 61 00 
  40e8d3:	49 81 fc e0 a5 61 00 	cmp    $0x61a5e0,%r12
  40e8da:	74 13                	je     40e8ef <__sprintf_chk@plt+0xc05f>
  40e8dc:	4c 89 e7             	mov    %r12,%rdi
  40e8df:	e8 0c 39 ff ff       	callq  4021f0 <free@plt>
  40e8e4:	48 c7 05 e9 bc 20 00 	movq   $0x61a5e0,0x20bce9(%rip)        # 61a5d8 <_fini@@Base+0x2086dc>
  40e8eb:	e0 a5 61 00 
  40e8ef:	5b                   	pop    %rbx
  40e8f0:	5d                   	pop    %rbp
  40e8f1:	c7 05 f5 bc 20 00 01 	movl   $0x1,0x20bcf5(%rip)        # 61a5f0 <_fini@@Base+0x2086f4>
  40e8f8:	00 00 00 
  40e8fb:	41 5c                	pop    %r12
  40e8fd:	c3                   	retq   
  40e8fe:	66 90                	xchg   %ax,%ax
  40e900:	b9 20 b3 61 00       	mov    $0x61b320,%ecx
  40e905:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40e90c:	e9 3f fb ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40e911:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e918:	0f 1f 84 00 00 00 00 
  40e91f:	00 
  40e920:	b9 20 b3 61 00       	mov    $0x61b320,%ecx
  40e925:	e9 26 fb ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40e92a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40e930:	48 89 fe             	mov    %rdi,%rsi
  40e933:	b9 20 b3 61 00       	mov    $0x61b320,%ecx
  40e938:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40e93f:	31 ff                	xor    %edi,%edi
  40e941:	e9 0a fb ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40e946:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40e94d:	00 00 00 
  40e950:	48 89 f2             	mov    %rsi,%rdx
  40e953:	b9 20 b3 61 00       	mov    $0x61b320,%ecx
  40e958:	48 89 fe             	mov    %rdi,%rsi
  40e95b:	31 ff                	xor    %edi,%edi
  40e95d:	e9 ee fa ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40e962:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e969:	1f 84 00 00 00 00 00 
  40e970:	55                   	push   %rbp
  40e971:	48 89 d5             	mov    %rdx,%rbp
  40e974:	53                   	push   %rbx
  40e975:	89 fb                	mov    %edi,%ebx
  40e977:	48 83 ec 48          	sub    $0x48,%rsp
  40e97b:	48 89 e7             	mov    %rsp,%rdi
  40e97e:	e8 bd ed ff ff       	callq  40d740 <__sprintf_chk@plt+0xaeb0>
  40e983:	48 89 e1             	mov    %rsp,%rcx
  40e986:	48 89 ee             	mov    %rbp,%rsi
  40e989:	89 df                	mov    %ebx,%edi
  40e98b:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40e992:	e8 b9 fa ff ff       	callq  40e450 <__sprintf_chk@plt+0xbbc0>
  40e997:	48 83 c4 48          	add    $0x48,%rsp
  40e99b:	5b                   	pop    %rbx
  40e99c:	5d                   	pop    %rbp
  40e99d:	c3                   	retq   
  40e99e:	66 90                	xchg   %ax,%ax
  40e9a0:	41 54                	push   %r12
  40e9a2:	49 89 cc             	mov    %rcx,%r12
  40e9a5:	55                   	push   %rbp
  40e9a6:	48 89 d5             	mov    %rdx,%rbp
  40e9a9:	53                   	push   %rbx
  40e9aa:	89 fb                	mov    %edi,%ebx
  40e9ac:	48 83 ec 40          	sub    $0x40,%rsp
  40e9b0:	48 89 e7             	mov    %rsp,%rdi
  40e9b3:	e8 88 ed ff ff       	callq  40d740 <__sprintf_chk@plt+0xaeb0>
  40e9b8:	48 89 e1             	mov    %rsp,%rcx
  40e9bb:	4c 89 e2             	mov    %r12,%rdx
  40e9be:	48 89 ee             	mov    %rbp,%rsi
  40e9c1:	89 df                	mov    %ebx,%edi
  40e9c3:	e8 88 fa ff ff       	callq  40e450 <__sprintf_chk@plt+0xbbc0>
  40e9c8:	48 83 c4 40          	add    $0x40,%rsp
  40e9cc:	5b                   	pop    %rbx
  40e9cd:	5d                   	pop    %rbp
  40e9ce:	41 5c                	pop    %r12
  40e9d0:	c3                   	retq   
  40e9d1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40e9d8:	0f 1f 84 00 00 00 00 
  40e9df:	00 
  40e9e0:	48 89 f2             	mov    %rsi,%rdx
  40e9e3:	89 fe                	mov    %edi,%esi
  40e9e5:	31 ff                	xor    %edi,%edi
  40e9e7:	e9 84 ff ff ff       	jmpq   40e970 <__sprintf_chk@plt+0xc0e0>
  40e9ec:	0f 1f 40 00          	nopl   0x0(%rax)
  40e9f0:	48 89 d1             	mov    %rdx,%rcx
  40e9f3:	48 89 f2             	mov    %rsi,%rdx
  40e9f6:	89 fe                	mov    %edi,%esi
  40e9f8:	31 ff                	xor    %edi,%edi
  40e9fa:	e9 a1 ff ff ff       	jmpq   40e9a0 <__sprintf_chk@plt+0xc110>
  40e9ff:	90                   	nop
  40ea00:	48 83 ec 48          	sub    $0x48,%rsp
  40ea04:	48 8b 05 15 c9 20 00 	mov    0x20c915(%rip),%rax        # 61b320 <stderr@@GLIBC_2.2.5+0xcd0>
  40ea0b:	41 89 d0             	mov    %edx,%r8d
  40ea0e:	41 c0 e8 05          	shr    $0x5,%r8b
  40ea12:	89 d1                	mov    %edx,%ecx
  40ea14:	48 89 f2             	mov    %rsi,%rdx
  40ea17:	45 0f b6 c0          	movzbl %r8b,%r8d
  40ea1b:	83 e1 1f             	and    $0x1f,%ecx
  40ea1e:	48 89 fe             	mov    %rdi,%rsi
  40ea21:	48 89 04 24          	mov    %rax,(%rsp)
  40ea25:	48 8b 05 fc c8 20 00 	mov    0x20c8fc(%rip),%rax        # 61b328 <stderr@@GLIBC_2.2.5+0xcd8>
  40ea2c:	31 ff                	xor    %edi,%edi
  40ea2e:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40ea33:	48 8b 05 f6 c8 20 00 	mov    0x20c8f6(%rip),%rax        # 61b330 <stderr@@GLIBC_2.2.5+0xce0>
  40ea3a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40ea3f:	48 8b 05 f2 c8 20 00 	mov    0x20c8f2(%rip),%rax        # 61b338 <stderr@@GLIBC_2.2.5+0xce8>
  40ea46:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40ea4b:	48 8b 05 ee c8 20 00 	mov    0x20c8ee(%rip),%rax        # 61b340 <stderr@@GLIBC_2.2.5+0xcf0>
  40ea52:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40ea57:	48 8b 05 ea c8 20 00 	mov    0x20c8ea(%rip),%rax        # 61b348 <stderr@@GLIBC_2.2.5+0xcf8>
  40ea5e:	46 8b 4c 84 08       	mov    0x8(%rsp,%r8,4),%r9d
  40ea63:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40ea68:	48 8b 05 e1 c8 20 00 	mov    0x20c8e1(%rip),%rax        # 61b350 <stderr@@GLIBC_2.2.5+0xd00>
  40ea6f:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  40ea74:	44 89 c8             	mov    %r9d,%eax
  40ea77:	d3 e8                	shr    %cl,%eax
  40ea79:	83 f0 01             	xor    $0x1,%eax
  40ea7c:	83 e0 01             	and    $0x1,%eax
  40ea7f:	d3 e0                	shl    %cl,%eax
  40ea81:	48 89 e1             	mov    %rsp,%rcx
  40ea84:	44 31 c8             	xor    %r9d,%eax
  40ea87:	42 89 44 84 08       	mov    %eax,0x8(%rsp,%r8,4)
  40ea8c:	e8 bf f9 ff ff       	callq  40e450 <__sprintf_chk@plt+0xbbc0>
  40ea91:	48 83 c4 48          	add    $0x48,%rsp
  40ea95:	c3                   	retq   
  40ea96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40ea9d:	00 00 00 
  40eaa0:	40 0f be d6          	movsbl %sil,%edx
  40eaa4:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  40eaab:	e9 50 ff ff ff       	jmpq   40ea00 <__sprintf_chk@plt+0xc170>
  40eab0:	ba 3a 00 00 00       	mov    $0x3a,%edx
  40eab5:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  40eabc:	e9 3f ff ff ff       	jmpq   40ea00 <__sprintf_chk@plt+0xc170>
  40eac1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40eac8:	0f 1f 84 00 00 00 00 
  40eacf:	00 
  40ead0:	ba 3a 00 00 00       	mov    $0x3a,%edx
  40ead5:	e9 26 ff ff ff       	jmpq   40ea00 <__sprintf_chk@plt+0xc170>
  40eada:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40eae0:	41 54                	push   %r12
  40eae2:	4d 89 c4             	mov    %r8,%r12
  40eae5:	55                   	push   %rbp
  40eae6:	48 89 cd             	mov    %rcx,%rbp
  40eae9:	53                   	push   %rbx
  40eaea:	89 fb                	mov    %edi,%ebx
  40eaec:	48 83 ec 40          	sub    $0x40,%rsp
  40eaf0:	48 8b 05 29 c8 20 00 	mov    0x20c829(%rip),%rax        # 61b320 <stderr@@GLIBC_2.2.5+0xcd0>
  40eaf7:	48 89 e7             	mov    %rsp,%rdi
  40eafa:	48 89 04 24          	mov    %rax,(%rsp)
  40eafe:	48 8b 05 23 c8 20 00 	mov    0x20c823(%rip),%rax        # 61b328 <stderr@@GLIBC_2.2.5+0xcd8>
  40eb05:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  40eb0a:	48 8b 05 1f c8 20 00 	mov    0x20c81f(%rip),%rax        # 61b330 <stderr@@GLIBC_2.2.5+0xce0>
  40eb11:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  40eb16:	48 8b 05 1b c8 20 00 	mov    0x20c81b(%rip),%rax        # 61b338 <stderr@@GLIBC_2.2.5+0xce8>
  40eb1d:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40eb22:	48 8b 05 17 c8 20 00 	mov    0x20c817(%rip),%rax        # 61b340 <stderr@@GLIBC_2.2.5+0xcf0>
  40eb29:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  40eb2e:	48 8b 05 13 c8 20 00 	mov    0x20c813(%rip),%rax        # 61b348 <stderr@@GLIBC_2.2.5+0xcf8>
  40eb35:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  40eb3a:	48 8b 05 0f c8 20 00 	mov    0x20c80f(%rip),%rax        # 61b350 <stderr@@GLIBC_2.2.5+0xd00>
  40eb41:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  40eb46:	e8 65 fb ff ff       	callq  40e6b0 <__sprintf_chk@plt+0xbe20>
  40eb4b:	48 89 e1             	mov    %rsp,%rcx
  40eb4e:	4c 89 e2             	mov    %r12,%rdx
  40eb51:	48 89 ee             	mov    %rbp,%rsi
  40eb54:	89 df                	mov    %ebx,%edi
  40eb56:	e8 f5 f8 ff ff       	callq  40e450 <__sprintf_chk@plt+0xbbc0>
  40eb5b:	48 83 c4 40          	add    $0x40,%rsp
  40eb5f:	5b                   	pop    %rbx
  40eb60:	5d                   	pop    %rbp
  40eb61:	41 5c                	pop    %r12
  40eb63:	c3                   	retq   
  40eb64:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40eb6b:	00 00 00 00 00 
  40eb70:	49 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%r8
  40eb77:	e9 64 ff ff ff       	jmpq   40eae0 <__sprintf_chk@plt+0xc250>
  40eb7c:	0f 1f 40 00          	nopl   0x0(%rax)
  40eb80:	48 89 d1             	mov    %rdx,%rcx
  40eb83:	49 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%r8
  40eb8a:	48 89 f2             	mov    %rsi,%rdx
  40eb8d:	48 89 fe             	mov    %rdi,%rsi
  40eb90:	31 ff                	xor    %edi,%edi
  40eb92:	e9 49 ff ff ff       	jmpq   40eae0 <__sprintf_chk@plt+0xc250>
  40eb97:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40eb9e:	00 00 
  40eba0:	49 89 c8             	mov    %rcx,%r8
  40eba3:	48 89 d1             	mov    %rdx,%rcx
  40eba6:	48 89 f2             	mov    %rsi,%rdx
  40eba9:	48 89 fe             	mov    %rdi,%rsi
  40ebac:	31 ff                	xor    %edi,%edi
  40ebae:	e9 2d ff ff ff       	jmpq   40eae0 <__sprintf_chk@plt+0xc250>
  40ebb3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40ebba:	84 00 00 00 00 00 
  40ebc0:	b9 a0 a5 61 00       	mov    $0x61a5a0,%ecx
  40ebc5:	e9 86 f8 ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40ebca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ebd0:	48 89 f2             	mov    %rsi,%rdx
  40ebd3:	b9 a0 a5 61 00       	mov    $0x61a5a0,%ecx
  40ebd8:	48 89 fe             	mov    %rdi,%rsi
  40ebdb:	31 ff                	xor    %edi,%edi
  40ebdd:	e9 6e f8 ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40ebe2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40ebe9:	1f 84 00 00 00 00 00 
  40ebf0:	b9 a0 a5 61 00       	mov    $0x61a5a0,%ecx
  40ebf5:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40ebfc:	e9 4f f8 ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40ec01:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40ec08:	0f 1f 84 00 00 00 00 
  40ec0f:	00 
  40ec10:	48 89 fe             	mov    %rdi,%rsi
  40ec13:	b9 a0 a5 61 00       	mov    $0x61a5a0,%ecx
  40ec18:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40ec1f:	31 ff                	xor    %edi,%edi
  40ec21:	e9 2a f8 ff ff       	jmpq   40e450 <__sprintf_chk@plt+0xbbc0>
  40ec26:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40ec2d:	00 00 00 
  40ec30:	41 54                	push   %r12
  40ec32:	48 85 d2             	test   %rdx,%rdx
  40ec35:	55                   	push   %rbp
  40ec36:	48 89 fd             	mov    %rdi,%rbp
  40ec39:	53                   	push   %rbx
  40ec3a:	48 8d 5a ff          	lea    -0x1(%rdx),%rbx
  40ec3e:	74 29                	je     40ec69 <__sprintf_chk@plt+0xc3d9>
  40ec40:	49 89 f4             	mov    %rsi,%r12
  40ec43:	e8 28 3c ff ff       	callq  402870 <__ctype_tolower_loc@plt>
  40ec48:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40ec4f:	00 
  40ec50:	41 0f b6 0c 1c       	movzbl (%r12,%rbx,1),%ecx
  40ec55:	48 8b 10             	mov    (%rax),%rdx
  40ec58:	8b 14 8a             	mov    (%rdx,%rcx,4),%edx
  40ec5b:	88 54 1d 00          	mov    %dl,0x0(%rbp,%rbx,1)
  40ec5f:	48 83 eb 01          	sub    $0x1,%rbx
  40ec63:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  40ec67:	75 e7                	jne    40ec50 <__sprintf_chk@plt+0xc3c0>
  40ec69:	5b                   	pop    %rbx
  40ec6a:	48 89 e8             	mov    %rbp,%rax
  40ec6d:	5d                   	pop    %rbp
  40ec6e:	41 5c                	pop    %r12
  40ec70:	c3                   	retq   
  40ec71:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40ec78:	0f 1f 84 00 00 00 00 
  40ec7f:	00 
  40ec80:	41 54                	push   %r12
  40ec82:	48 85 d2             	test   %rdx,%rdx
  40ec85:	55                   	push   %rbp
  40ec86:	48 89 fd             	mov    %rdi,%rbp
  40ec89:	53                   	push   %rbx
  40ec8a:	48 8d 5a ff          	lea    -0x1(%rdx),%rbx
  40ec8e:	74 29                	je     40ecb9 <__sprintf_chk@plt+0xc429>
  40ec90:	49 89 f4             	mov    %rsi,%r12
  40ec93:	e8 08 35 ff ff       	callq  4021a0 <__ctype_toupper_loc@plt>
  40ec98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40ec9f:	00 
  40eca0:	41 0f b6 0c 1c       	movzbl (%r12,%rbx,1),%ecx
  40eca5:	48 8b 10             	mov    (%rax),%rdx
  40eca8:	8b 14 8a             	mov    (%rdx,%rcx,4),%edx
  40ecab:	88 54 1d 00          	mov    %dl,0x0(%rbp,%rbx,1)
  40ecaf:	48 83 eb 01          	sub    $0x1,%rbx
  40ecb3:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  40ecb7:	75 e7                	jne    40eca0 <__sprintf_chk@plt+0xc410>
  40ecb9:	5b                   	pop    %rbx
  40ecba:	48 89 e8             	mov    %rbp,%rax
  40ecbd:	5d                   	pop    %rbp
  40ecbe:	41 5c                	pop    %r12
  40ecc0:	c3                   	retq   
  40ecc1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  40ecc8:	0f 1f 84 00 00 00 00 
  40eccf:	00 
  40ecd0:	41 57                	push   %r15
  40ecd2:	49 89 d2             	mov    %rdx,%r10
  40ecd5:	41 56                	push   %r14
  40ecd7:	41 55                	push   %r13
  40ecd9:	41 54                	push   %r12
  40ecdb:	49 89 f4             	mov    %rsi,%r12
  40ecde:	55                   	push   %rbp
  40ecdf:	53                   	push   %rbx
  40ece0:	48 89 cb             	mov    %rcx,%rbx
  40ece3:	48 81 ec d8 04 00 00 	sub    $0x4d8,%rsp
  40ecea:	49 8b 40 30          	mov    0x30(%r8),%rax
  40ecee:	64 48 8b 34 25 28 00 	mov    %fs:0x28,%rsi
  40ecf5:	00 00 
  40ecf7:	48 89 b4 24 c8 04 00 	mov    %rsi,0x4c8(%rsp)
  40ecfe:	00 
  40ecff:	31 f6                	xor    %esi,%esi
  40ed01:	41 8b 70 08          	mov    0x8(%r8),%esi
  40ed05:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
  40ed0a:	44 89 4c 24 44       	mov    %r9d,0x44(%rsp)
  40ed0f:	40 88 7c 24 0f       	mov    %dil,0xf(%rsp)
  40ed14:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  40ed19:	83 fe 0c             	cmp    $0xc,%esi
  40ed1c:	89 74 24 40          	mov    %esi,0x40(%rsp)
  40ed20:	0f 8e ba 01 00 00    	jle    40eee0 <__sprintf_chk@plt+0xc650>
  40ed26:	83 6c 24 40 0c       	subl   $0xc,0x40(%rsp)
  40ed2b:	0f b6 03             	movzbl (%rbx),%eax
  40ed2e:	45 31 ed             	xor    %r13d,%r13d
  40ed31:	84 c0                	test   %al,%al
  40ed33:	0f 84 1a 03 00 00    	je     40f053 <__sprintf_chk@plt+0xc7c3>
  40ed39:	48 8d b4 24 c1 00 00 	lea    0xc1(%rsp),%rsi
  40ed40:	00 
  40ed41:	4d 89 d6             	mov    %r10,%r14
  40ed44:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
  40ed49:	eb 3a                	jmp    40ed85 <__sprintf_chk@plt+0xc4f5>
  40ed4b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ed50:	4c 89 f2             	mov    %r14,%rdx
  40ed53:	4c 29 ea             	sub    %r13,%rdx
  40ed56:	48 83 fa 01          	cmp    $0x1,%rdx
  40ed5a:	0f 86 38 01 00 00    	jbe    40ee98 <__sprintf_chk@plt+0xc608>
  40ed60:	4d 85 e4             	test   %r12,%r12
  40ed63:	74 08                	je     40ed6d <__sprintf_chk@plt+0xc4dd>
  40ed65:	41 88 04 24          	mov    %al,(%r12)
  40ed69:	49 83 c4 01          	add    $0x1,%r12
  40ed6d:	49 83 c5 01          	add    $0x1,%r13
  40ed71:	49 89 d8             	mov    %rbx,%r8
  40ed74:	41 0f b6 40 01       	movzbl 0x1(%r8),%eax
  40ed79:	49 8d 58 01          	lea    0x1(%r8),%rbx
  40ed7d:	84 c0                	test   %al,%al
  40ed7f:	0f 84 cb 02 00 00    	je     40f050 <__sprintf_chk@plt+0xc7c0>
  40ed85:	3c 25                	cmp    $0x25,%al
  40ed87:	75 c7                	jne    40ed50 <__sprintf_chk@plt+0xc4c0>
  40ed89:	44 0f b6 4c 24 0f    	movzbl 0xf(%rsp),%r9d
  40ed8f:	31 c0                	xor    %eax,%eax
  40ed91:	45 31 db             	xor    %r11d,%r11d
  40ed94:	48 83 c3 01          	add    $0x1,%rbx
  40ed98:	0f b6 3b             	movzbl (%rbx),%edi
  40ed9b:	40 80 ff 30          	cmp    $0x30,%dil
  40ed9f:	74 1f                	je     40edc0 <__sprintf_chk@plt+0xc530>
  40eda1:	7f 2d                	jg     40edd0 <__sprintf_chk@plt+0xc540>
  40eda3:	40 80 ff 23          	cmp    $0x23,%dil
  40eda7:	75 3f                	jne    40ede8 <__sprintf_chk@plt+0xc558>
  40eda9:	48 83 c3 01          	add    $0x1,%rbx
  40edad:	0f b6 3b             	movzbl (%rbx),%edi
  40edb0:	b8 01 00 00 00       	mov    $0x1,%eax
  40edb5:	40 80 ff 30          	cmp    $0x30,%dil
  40edb9:	75 e6                	jne    40eda1 <__sprintf_chk@plt+0xc511>
  40edbb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40edc0:	44 0f be df          	movsbl %dil,%r11d
  40edc4:	eb ce                	jmp    40ed94 <__sprintf_chk@plt+0xc504>
  40edc6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40edcd:	00 00 00 
  40edd0:	40 80 ff 5e          	cmp    $0x5e,%dil
  40edd4:	0f 85 96 00 00 00    	jne    40ee70 <__sprintf_chk@plt+0xc5e0>
  40edda:	41 b9 01 00 00 00    	mov    $0x1,%r9d
  40ede0:	eb b2                	jmp    40ed94 <__sprintf_chk@plt+0xc504>
  40ede2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40ede8:	40 80 ff 2d          	cmp    $0x2d,%dil
  40edec:	74 d2                	je     40edc0 <__sprintf_chk@plt+0xc530>
  40edee:	40 0f be d7          	movsbl %dil,%edx
  40edf2:	bd ff ff ff ff       	mov    $0xffffffff,%ebp
  40edf7:	83 ea 30             	sub    $0x30,%edx
  40edfa:	83 fa 09             	cmp    $0x9,%edx
  40edfd:	77 41                	ja     40ee40 <__sprintf_chk@plt+0xc5b0>
  40edff:	31 ed                	xor    %ebp,%ebp
  40ee01:	eb 25                	jmp    40ee28 <__sprintf_chk@plt+0xc598>
  40ee03:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ee08:	0f be 13             	movsbl (%rbx),%edx
  40ee0b:	74 7b                	je     40ee88 <__sprintf_chk@plt+0xc5f8>
  40ee0d:	8d 4c ad 00          	lea    0x0(%rbp,%rbp,4),%ecx
  40ee11:	8d 6c 4a d0          	lea    -0x30(%rdx,%rcx,2),%ebp
  40ee15:	48 83 c3 01          	add    $0x1,%rbx
  40ee19:	0f b6 3b             	movzbl (%rbx),%edi
  40ee1c:	40 0f be d7          	movsbl %dil,%edx
  40ee20:	83 ea 30             	sub    $0x30,%edx
  40ee23:	83 fa 09             	cmp    $0x9,%edx
  40ee26:	77 18                	ja     40ee40 <__sprintf_chk@plt+0xc5b0>
  40ee28:	81 fd cc cc cc 0c    	cmp    $0xccccccc,%ebp
  40ee2e:	7e d8                	jle    40ee08 <__sprintf_chk@plt+0xc578>
  40ee30:	bd ff ff ff 7f       	mov    $0x7fffffff,%ebp
  40ee35:	eb de                	jmp    40ee15 <__sprintf_chk@plt+0xc585>
  40ee37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40ee3e:	00 00 
  40ee40:	40 80 ff 45          	cmp    $0x45,%dil
  40ee44:	0f 84 7e 00 00 00    	je     40eec8 <__sprintf_chk@plt+0xc638>
  40ee4a:	31 c9                	xor    %ecx,%ecx
  40ee4c:	40 80 ff 4f          	cmp    $0x4f,%dil
  40ee50:	74 76                	je     40eec8 <__sprintf_chk@plt+0xc638>
  40ee52:	40 80 ff 7a          	cmp    $0x7a,%dil
  40ee56:	40 0f be f7          	movsbl %dil,%esi
  40ee5a:	0f 87 ed 10 00 00    	ja     40ff4d <__sprintf_chk@plt+0xd6bd>
  40ee60:	40 0f b6 d7          	movzbl %dil,%edx
  40ee64:	ff 24 d5 e8 64 41 00 	jmpq   *0x4164e8(,%rdx,8)
  40ee6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ee70:	40 80 ff 5f          	cmp    $0x5f,%dil
  40ee74:	0f 85 74 ff ff ff    	jne    40edee <__sprintf_chk@plt+0xc55e>
  40ee7a:	44 0f be df          	movsbl %dil,%r11d
  40ee7e:	e9 11 ff ff ff       	jmpq   40ed94 <__sprintf_chk@plt+0xc504>
  40ee83:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ee88:	80 fa 37             	cmp    $0x37,%dl
  40ee8b:	7e 80                	jle    40ee0d <__sprintf_chk@plt+0xc57d>
  40ee8d:	bd ff ff ff 7f       	mov    $0x7fffffff,%ebp
  40ee92:	eb 81                	jmp    40ee15 <__sprintf_chk@plt+0xc585>
  40ee94:	0f 1f 40 00          	nopl   0x0(%rax)
  40ee98:	31 c0                	xor    %eax,%eax
  40ee9a:	48 8b b4 24 c8 04 00 	mov    0x4c8(%rsp),%rsi
  40eea1:	00 
  40eea2:	64 48 33 34 25 28 00 	xor    %fs:0x28,%rsi
  40eea9:	00 00 
  40eeab:	0f 85 39 17 00 00    	jne    4105ea <__sprintf_chk@plt+0xdd5a>
  40eeb1:	48 81 c4 d8 04 00 00 	add    $0x4d8,%rsp
  40eeb8:	5b                   	pop    %rbx
  40eeb9:	5d                   	pop    %rbp
  40eeba:	41 5c                	pop    %r12
  40eebc:	41 5d                	pop    %r13
  40eebe:	41 5e                	pop    %r14
  40eec0:	41 5f                	pop    %r15
  40eec2:	c3                   	retq   
  40eec3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40eec8:	40 0f be cf          	movsbl %dil,%ecx
  40eecc:	48 83 c3 01          	add    $0x1,%rbx
  40eed0:	0f b6 3b             	movzbl (%rbx),%edi
  40eed3:	e9 7a ff ff ff       	jmpq   40ee52 <__sprintf_chk@plt+0xc5c2>
  40eed8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  40eedf:	00 
  40eee0:	8b 74 24 40          	mov    0x40(%rsp),%esi
  40eee4:	b8 0c 00 00 00       	mov    $0xc,%eax
  40eee9:	85 f6                	test   %esi,%esi
  40eeeb:	0f 45 c6             	cmovne %esi,%eax
  40eeee:	89 44 24 40          	mov    %eax,0x40(%rsp)
  40eef2:	e9 34 fe ff ff       	jmpq   40ed2b <__sprintf_chk@plt+0xc49b>
  40eef7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40eefe:	00 00 
  40ef00:	83 f9 4f             	cmp    $0x4f,%ecx
  40ef03:	0f 84 af 01 00 00    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40ef09:	45 31 ff             	xor    %r15d,%r15d
  40ef0c:	85 c9                	test   %ecx,%ecx
  40ef0e:	c6 84 24 b0 00 00 00 	movb   $0x20,0xb0(%rsp)
  40ef15:	20 
  40ef16:	c6 84 24 b1 00 00 00 	movb   $0x25,0xb1(%rsp)
  40ef1d:	25 
  40ef1e:	0f 85 cb 16 00 00    	jne    4105ef <__sprintf_chk@plt+0xdd5f>
  40ef24:	48 8d 84 24 b2 00 00 	lea    0xb2(%rsp),%rax
  40ef2b:	00 
  40ef2c:	49 89 d8             	mov    %rbx,%r8
  40ef2f:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
  40ef36:	00 
  40ef37:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
  40ef3c:	40 88 38             	mov    %dil,(%rax)
  40ef3f:	48 8d 94 24 b0 00 00 	lea    0xb0(%rsp),%rdx
  40ef46:	00 
  40ef47:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  40ef4e:	00 
  40ef4f:	c6 40 01 00          	movb   $0x0,0x1(%rax)
  40ef53:	be 00 04 00 00       	mov    $0x400,%esi
  40ef58:	44 89 4c 24 38       	mov    %r9d,0x38(%rsp)
  40ef5d:	44 89 5c 24 30       	mov    %r11d,0x30(%rsp)
  40ef62:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
  40ef67:	e8 d4 37 ff ff       	callq  402740 <strftime@plt>
  40ef6c:	48 85 c0             	test   %rax,%rax
  40ef6f:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
  40ef74:	44 8b 5c 24 30       	mov    0x30(%rsp),%r11d
  40ef79:	44 8b 4c 24 38       	mov    0x38(%rsp),%r9d
  40ef7e:	0f 84 f0 fd ff ff    	je     40ed74 <__sprintf_chk@plt+0xc4e4>
  40ef84:	48 8d 58 ff          	lea    -0x1(%rax),%rbx
  40ef88:	31 c0                	xor    %eax,%eax
  40ef8a:	85 ed                	test   %ebp,%ebp
  40ef8c:	0f 49 c5             	cmovns %ebp,%eax
  40ef8f:	4c 89 f2             	mov    %r14,%rdx
  40ef92:	48 98                	cltq   
  40ef94:	48 89 d9             	mov    %rbx,%rcx
  40ef97:	48 39 d8             	cmp    %rbx,%rax
  40ef9a:	48 0f 43 c8          	cmovae %rax,%rcx
  40ef9e:	4c 29 ea             	sub    %r13,%rdx
  40efa1:	48 39 d1             	cmp    %rdx,%rcx
  40efa4:	0f 83 ee fe ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40efaa:	4d 85 e4             	test   %r12,%r12
  40efad:	0f 84 87 00 00 00    	je     40f03a <__sprintf_chk@plt+0xc7aa>
  40efb3:	48 39 c3             	cmp    %rax,%rbx
  40efb6:	73 49                	jae    40f001 <__sprintf_chk@plt+0xc771>
  40efb8:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40efbc:	85 c0                	test   %eax,%eax
  40efbe:	75 41                	jne    40f001 <__sprintf_chk@plt+0xc771>
  40efc0:	48 63 ed             	movslq %ebp,%rbp
  40efc3:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  40efc8:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
  40efcd:	48 29 dd             	sub    %rbx,%rbp
  40efd0:	41 83 fb 30          	cmp    $0x30,%r11d
  40efd4:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40efd9:	48 89 ea             	mov    %rbp,%rdx
  40efdc:	0f 84 57 12 00 00    	je     410239 <__sprintf_chk@plt+0xd9a9>
  40efe2:	4c 89 e7             	mov    %r12,%rdi
  40efe5:	be 20 00 00 00       	mov    $0x20,%esi
  40efea:	49 01 ec             	add    %rbp,%r12
  40efed:	e8 8e 34 ff ff       	callq  402480 <memset@plt>
  40eff2:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  40eff7:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40effc:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f001:	45 84 ff             	test   %r15b,%r15b
  40f004:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
  40f009:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40f00e:	48 89 da             	mov    %rbx,%rdx
  40f011:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
  40f016:	4c 89 e7             	mov    %r12,%rdi
  40f019:	0f 85 8e 01 00 00    	jne    40f1ad <__sprintf_chk@plt+0xc91d>
  40f01f:	45 84 c9             	test   %r9b,%r9b
  40f022:	0f 84 71 01 00 00    	je     40f199 <__sprintf_chk@plt+0xc909>
  40f028:	e8 53 fc ff ff       	callq  40ec80 <__sprintf_chk@plt+0xc3f0>
  40f02d:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f032:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40f037:	49 01 dc             	add    %rbx,%r12
  40f03a:	41 0f b6 40 01       	movzbl 0x1(%r8),%eax
  40f03f:	49 01 cd             	add    %rcx,%r13
  40f042:	49 8d 58 01          	lea    0x1(%r8),%rbx
  40f046:	84 c0                	test   %al,%al
  40f048:	0f 85 37 fd ff ff    	jne    40ed85 <__sprintf_chk@plt+0xc4f5>
  40f04e:	66 90                	xchg   %ax,%ax
  40f050:	4d 89 f2             	mov    %r14,%r10
  40f053:	4d 85 e4             	test   %r12,%r12
  40f056:	0f 84 35 01 00 00    	je     40f191 <__sprintf_chk@plt+0xc901>
  40f05c:	4d 85 d2             	test   %r10,%r10
  40f05f:	0f 84 2c 01 00 00    	je     40f191 <__sprintf_chk@plt+0xc901>
  40f065:	41 c6 04 24 00       	movb   $0x0,(%r12)
  40f06a:	4c 89 e8             	mov    %r13,%rax
  40f06d:	e9 28 fe ff ff       	jmpq   40ee9a <__sprintf_chk@plt+0xc60a>
  40f072:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  40f078:	85 c9                	test   %ecx,%ecx
  40f07a:	75 3c                	jne    40f0b8 <__sprintf_chk@plt+0xc828>
  40f07c:	84 c0                	test   %al,%al
  40f07e:	b8 01 00 00 00       	mov    $0x1,%eax
  40f083:	44 0f 45 c8          	cmovne %eax,%r9d
  40f087:	c6 84 24 b0 00 00 00 	movb   $0x20,0xb0(%rsp)
  40f08e:	20 
  40f08f:	c6 84 24 b1 00 00 00 	movb   $0x25,0xb1(%rsp)
  40f096:	25 
  40f097:	49 89 d8             	mov    %rbx,%r8
  40f09a:	45 31 ff             	xor    %r15d,%r15d
  40f09d:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
  40f0a4:	00 
  40f0a5:	48 8d 84 24 b2 00 00 	lea    0xb2(%rsp),%rax
  40f0ac:	00 
  40f0ad:	e9 85 fe ff ff       	jmpq   40ef37 <__sprintf_chk@plt+0xc6a7>
  40f0b2:	4c 89 c3             	mov    %r8,%rbx
  40f0b5:	0f 1f 00             	nopl   (%rax)
  40f0b8:	48 8d 43 ff          	lea    -0x1(%rbx),%rax
  40f0bc:	b9 01 00 00 00       	mov    $0x1,%ecx
  40f0c1:	49 89 c7             	mov    %rax,%r15
  40f0c4:	48 8d 40 ff          	lea    -0x1(%rax),%rax
  40f0c8:	83 c1 01             	add    $0x1,%ecx
  40f0cb:	80 78 01 25          	cmpb   $0x25,0x1(%rax)
  40f0cf:	75 f0                	jne    40f0c1 <__sprintf_chk@plt+0xc831>
  40f0d1:	48 63 c9             	movslq %ecx,%rcx
  40f0d4:	49 89 d8             	mov    %rbx,%r8
  40f0d7:	31 c0                	xor    %eax,%eax
  40f0d9:	85 ed                	test   %ebp,%ebp
  40f0db:	4c 89 f2             	mov    %r14,%rdx
  40f0de:	0f 49 c5             	cmovns %ebp,%eax
  40f0e1:	48 98                	cltq   
  40f0e3:	48 39 c1             	cmp    %rax,%rcx
  40f0e6:	48 89 c3             	mov    %rax,%rbx
  40f0e9:	48 0f 43 d9          	cmovae %rcx,%rbx
  40f0ed:	4c 29 ea             	sub    %r13,%rdx
  40f0f0:	48 39 d3             	cmp    %rdx,%rbx
  40f0f3:	0f 83 9f fd ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40f0f9:	4d 85 e4             	test   %r12,%r12
  40f0fc:	74 70                	je     40f16e <__sprintf_chk@plt+0xc8de>
  40f0fe:	48 39 c8             	cmp    %rcx,%rax
  40f101:	76 41                	jbe    40f144 <__sprintf_chk@plt+0xc8b4>
  40f103:	48 63 ed             	movslq %ebp,%rbp
  40f106:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
  40f10b:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
  40f110:	48 29 cd             	sub    %rcx,%rbp
  40f113:	41 83 fb 30          	cmp    $0x30,%r11d
  40f117:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40f11c:	48 89 ea             	mov    %rbp,%rdx
  40f11f:	0f 84 9c 00 00 00    	je     40f1c1 <__sprintf_chk@plt+0xc931>
  40f125:	4c 89 e7             	mov    %r12,%rdi
  40f128:	be 20 00 00 00       	mov    $0x20,%esi
  40f12d:	49 01 ec             	add    %rbp,%r12
  40f130:	e8 4b 33 ff ff       	callq  402480 <memset@plt>
  40f135:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  40f13a:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40f13f:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f144:	45 84 c9             	test   %r9b,%r9b
  40f147:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
  40f14c:	48 89 ca             	mov    %rcx,%rdx
  40f14f:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
  40f154:	4c 89 fe             	mov    %r15,%rsi
  40f157:	4c 89 e7             	mov    %r12,%rdi
  40f15a:	74 24                	je     40f180 <__sprintf_chk@plt+0xc8f0>
  40f15c:	e8 1f fb ff ff       	callq  40ec80 <__sprintf_chk@plt+0xc3f0>
  40f161:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40f166:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
  40f16b:	49 01 cc             	add    %rcx,%r12
  40f16e:	49 01 dd             	add    %rbx,%r13
  40f171:	e9 fe fb ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  40f176:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40f17d:	00 00 00 
  40f180:	e8 3b 34 ff ff       	callq  4025c0 <memcpy@plt>
  40f185:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
  40f18a:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40f18f:	eb da                	jmp    40f16b <__sprintf_chk@plt+0xc8db>
  40f191:	4c 89 e8             	mov    %r13,%rax
  40f194:	e9 01 fd ff ff       	jmpq   40ee9a <__sprintf_chk@plt+0xc60a>
  40f199:	e8 22 34 ff ff       	callq  4025c0 <memcpy@plt>
  40f19e:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40f1a3:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f1a8:	e9 8a fe ff ff       	jmpq   40f037 <__sprintf_chk@plt+0xc7a7>
  40f1ad:	e8 7e fa ff ff       	callq  40ec30 <__sprintf_chk@plt+0xc3a0>
  40f1b2:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f1b7:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
  40f1bc:	e9 76 fe ff ff       	jmpq   40f037 <__sprintf_chk@plt+0xc7a7>
  40f1c1:	4c 89 e7             	mov    %r12,%rdi
  40f1c4:	be 30 00 00 00       	mov    $0x30,%esi
  40f1c9:	49 01 ec             	add    %rbp,%r12
  40f1cc:	e8 af 32 ff ff       	callq  402480 <memset@plt>
  40f1d1:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f1d6:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40f1db:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  40f1e0:	e9 5f ff ff ff       	jmpq   40f144 <__sprintf_chk@plt+0xc8b4>
  40f1e5:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f1ea:	ba 93 24 49 92       	mov    $0x92492493,%edx
  40f1ef:	c7 44 24 18 01 00 00 	movl   $0x1,0x18(%rsp)
  40f1f6:	00 
  40f1f7:	8b 40 18             	mov    0x18(%rax),%eax
  40f1fa:	44 8d 40 06          	lea    0x6(%rax),%r8d
  40f1fe:	44 89 c0             	mov    %r8d,%eax
  40f201:	f7 ea                	imul   %edx
  40f203:	44 89 c0             	mov    %r8d,%eax
  40f206:	c1 f8 1f             	sar    $0x1f,%eax
  40f209:	44 01 c2             	add    %r8d,%edx
  40f20c:	c1 fa 02             	sar    $0x2,%edx
  40f20f:	29 c2                	sub    %eax,%edx
  40f211:	8d 04 d5 00 00 00 00 	lea    0x0(,%rdx,8),%eax
  40f218:	29 d0                	sub    %edx,%eax
  40f21a:	41 29 c0             	sub    %eax,%r8d
  40f21d:	41 83 c0 01          	add    $0x1,%r8d
  40f221:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40f228:	44 89 c0             	mov    %r8d,%eax
  40f22b:	31 ff                	xor    %edi,%edi
  40f22d:	c6 44 24 28 00       	movb   $0x0,0x28(%rsp)
  40f232:	c1 e8 1f             	shr    $0x1f,%eax
  40f235:	41 89 c2             	mov    %eax,%r10d
  40f238:	45 84 d2             	test   %r10b,%r10b
  40f23b:	0f 85 d7 02 00 00    	jne    40f518 <__sprintf_chk@plt+0xcc88>
  40f241:	83 f9 4f             	cmp    $0x4f,%ecx
  40f244:	0f 85 ce 02 00 00    	jne    40f518 <__sprintf_chk@plt+0xcc88>
  40f24a:	c6 84 24 b0 00 00 00 	movb   $0x20,0xb0(%rsp)
  40f251:	20 
  40f252:	c6 84 24 b1 00 00 00 	movb   $0x25,0xb1(%rsp)
  40f259:	25 
  40f25a:	45 31 ff             	xor    %r15d,%r15d
  40f25d:	88 8c 24 b2 00 00 00 	mov    %cl,0xb2(%rsp)
  40f264:	89 f7                	mov    %esi,%edi
  40f266:	49 89 d8             	mov    %rbx,%r8
  40f269:	48 8d 84 24 b3 00 00 	lea    0xb3(%rsp),%rax
  40f270:	00 
  40f271:	e9 c1 fc ff ff       	jmpq   40ef37 <__sprintf_chk@plt+0xc6a7>
  40f276:	83 f9 45             	cmp    $0x45,%ecx
  40f279:	0f 84 39 fe ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f27f:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f284:	c6 44 24 28 00       	movb   $0x0,0x28(%rsp)
  40f289:	c7 44 24 18 03 00 00 	movl   $0x3,0x18(%rsp)
  40f290:	00 
  40f291:	44 8b 40 1c          	mov    0x1c(%rax),%r8d
  40f295:	41 83 f8 ff          	cmp    $0xffffffff,%r8d
  40f299:	41 0f 9c c2          	setl   %r10b
  40f29d:	41 83 c0 01          	add    $0x1,%r8d
  40f2a1:	31 ff                	xor    %edi,%edi
  40f2a3:	eb 93                	jmp    40f238 <__sprintf_chk@plt+0xc9a8>
  40f2a5:	83 f9 45             	cmp    $0x45,%ecx
  40f2a8:	0f 84 0a fe ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f2ae:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f2b3:	44 8b 40 0c          	mov    0xc(%rax),%r8d
  40f2b7:	41 83 fb 2d          	cmp    $0x2d,%r11d
  40f2bb:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f2c2:	00 
  40f2c3:	0f 84 5f ff ff ff    	je     40f228 <__sprintf_chk@plt+0xc998>
  40f2c9:	41 83 fb 30          	cmp    $0x30,%r11d
  40f2cd:	0f 84 55 ff ff ff    	je     40f228 <__sprintf_chk@plt+0xc998>
  40f2d3:	41 bb 5f 00 00 00    	mov    $0x5f,%r11d
  40f2d9:	e9 4a ff ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f2de:	83 f9 45             	cmp    $0x45,%ecx
  40f2e1:	0f 84 d1 fd ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f2e7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f2ec:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f2f3:	00 
  40f2f4:	44 8b 40 0c          	mov    0xc(%rax),%r8d
  40f2f8:	e9 2b ff ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f2fd:	45 31 ff             	xor    %r15d,%r15d
  40f300:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f305:	8b 78 20             	mov    0x20(%rax),%edi
  40f308:	85 ff                	test   %edi,%edi
  40f30a:	0f 88 92 0f 00 00    	js     4102a2 <__sprintf_chk@plt+0xda12>
  40f310:	4c 8b 50 28          	mov    0x28(%rax),%r10
  40f314:	ba c5 b3 a2 91       	mov    $0x91a2b3c5,%edx
  40f319:	44 89 d0             	mov    %r10d,%eax
  40f31c:	f7 ea                	imul   %edx
  40f31e:	44 89 d0             	mov    %r10d,%eax
  40f321:	c1 f8 1f             	sar    $0x1f,%eax
  40f324:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40f328:	46 8d 04 12          	lea    (%rdx,%r10,1),%r8d
  40f32c:	41 c1 f8 0b          	sar    $0xb,%r8d
  40f330:	41 29 c0             	sub    %eax,%r8d
  40f333:	b8 89 88 88 88       	mov    $0x88888889,%eax
  40f338:	41 f7 ea             	imul   %r10d
  40f33b:	b8 89 88 88 88       	mov    $0x88888889,%eax
  40f340:	42 8d 3c 12          	lea    (%rdx,%r10,1),%edi
  40f344:	c1 ff 05             	sar    $0x5,%edi
  40f347:	2b 7c 24 18          	sub    0x18(%rsp),%edi
  40f34b:	f7 ef                	imul   %edi
  40f34d:	8d 04 3a             	lea    (%rdx,%rdi,1),%eax
  40f350:	89 fa                	mov    %edi,%edx
  40f352:	c1 fa 1f             	sar    $0x1f,%edx
  40f355:	c1 f8 05             	sar    $0x5,%eax
  40f358:	29 d0                	sub    %edx,%eax
  40f35a:	ba 3c 00 00 00       	mov    $0x3c,%edx
  40f35f:	0f af c2             	imul   %edx,%eax
  40f362:	89 fa                	mov    %edi,%edx
  40f364:	29 c2                	sub    %eax,%edx
  40f366:	89 d0                	mov    %edx,%eax
  40f368:	ba 3c 00 00 00       	mov    $0x3c,%edx
  40f36d:	0f af fa             	imul   %edx,%edi
  40f370:	44 89 d2             	mov    %r10d,%edx
  40f373:	29 fa                	sub    %edi,%edx
  40f375:	49 83 ff 01          	cmp    $0x1,%r15
  40f379:	0f 84 d6 0f 00 00    	je     410355 <__sprintf_chk@plt+0xdac5>
  40f37f:	0f 82 6f 11 00 00    	jb     4104f4 <__sprintf_chk@plt+0xdc64>
  40f385:	49 83 ff 02          	cmp    $0x2,%r15
  40f389:	0f 84 44 0f 00 00    	je     4102d3 <__sprintf_chk@plt+0xda43>
  40f38f:	49 83 ff 03          	cmp    $0x3,%r15
  40f393:	0f 84 05 11 00 00    	je     41049e <__sprintf_chk@plt+0xdc0e>
  40f399:	0f b6 3b             	movzbl (%rbx),%edi
  40f39c:	49 89 d8             	mov    %rbx,%r8
  40f39f:	40 80 ff 25          	cmp    $0x25,%dil
  40f3a3:	0f 85 09 fd ff ff    	jne    40f0b2 <__sprintf_chk@plt+0xc822>
  40f3a9:	4d 89 c7             	mov    %r8,%r15
  40f3ac:	b9 01 00 00 00       	mov    $0x1,%ecx
  40f3b1:	e9 21 fd ff ff       	jmpq   40f0d7 <__sprintf_chk@plt+0xc847>
  40f3b6:	84 c0                	test   %al,%al
  40f3b8:	b8 01 00 00 00       	mov    $0x1,%eax
  40f3bd:	44 0f 45 c8          	cmovne %eax,%r9d
  40f3c1:	85 c9                	test   %ecx,%ecx
  40f3c3:	0f 85 ef fc ff ff    	jne    40f0b8 <__sprintf_chk@plt+0xc828>
  40f3c9:	e9 b9 fc ff ff       	jmpq   40f087 <__sprintf_chk@plt+0xc7f7>
  40f3ce:	83 f9 45             	cmp    $0x45,%ecx
  40f3d1:	0f 84 e1 fc ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f3d7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f3dc:	c7 44 24 18 01 00 00 	movl   $0x1,0x18(%rsp)
  40f3e3:	00 
  40f3e4:	44 8b 40 18          	mov    0x18(%rax),%r8d
  40f3e8:	e9 3b fe ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f3ed:	3c 01                	cmp    $0x1,%al
  40f3ef:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
  40f3f4:	19 c9                	sbb    %ecx,%ecx
  40f3f6:	83 c1 01             	add    $0x1,%ecx
  40f3f9:	84 c0                	test   %al,%al
  40f3fb:	b8 00 00 00 00       	mov    $0x0,%eax
  40f400:	44 0f 45 c8          	cmovne %eax,%r9d
  40f404:	48 85 ff             	test   %rdi,%rdi
  40f407:	0f 84 35 11 00 00    	je     410542 <__sprintf_chk@plt+0xdcb2>
  40f40d:	89 4c 24 30          	mov    %ecx,0x30(%rsp)
  40f411:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
  40f416:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
  40f41b:	e8 60 2f ff ff       	callq  402380 <strlen@plt>
  40f420:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
  40f425:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40f42a:	49 89 c7             	mov    %rax,%r15
  40f42d:	8b 4c 24 30          	mov    0x30(%rsp),%ecx
  40f431:	31 c0                	xor    %eax,%eax
  40f433:	85 ed                	test   %ebp,%ebp
  40f435:	4c 89 f2             	mov    %r14,%rdx
  40f438:	0f 49 c5             	cmovns %ebp,%eax
  40f43b:	48 98                	cltq   
  40f43d:	49 39 c7             	cmp    %rax,%r15
  40f440:	49 89 c0             	mov    %rax,%r8
  40f443:	4d 0f 43 c7          	cmovae %r15,%r8
  40f447:	4c 29 ea             	sub    %r13,%rdx
  40f44a:	49 39 d0             	cmp    %rdx,%r8
  40f44d:	0f 83 45 fa ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40f453:	4d 85 e4             	test   %r12,%r12
  40f456:	74 72                	je     40f4ca <__sprintf_chk@plt+0xcc3a>
  40f458:	4c 39 f8             	cmp    %r15,%rax
  40f45b:	76 3f                	jbe    40f49c <__sprintf_chk@plt+0xcc0c>
  40f45d:	48 63 ed             	movslq %ebp,%rbp
  40f460:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
  40f465:	89 4c 24 28          	mov    %ecx,0x28(%rsp)
  40f469:	4c 29 fd             	sub    %r15,%rbp
  40f46c:	41 83 fb 30          	cmp    $0x30,%r11d
  40f470:	44 89 4c 24 18       	mov    %r9d,0x18(%rsp)
  40f475:	48 89 ea             	mov    %rbp,%rdx
  40f478:	0f 84 ed 10 00 00    	je     41056b <__sprintf_chk@plt+0xdcdb>
  40f47e:	4c 89 e7             	mov    %r12,%rdi
  40f481:	be 20 00 00 00       	mov    $0x20,%esi
  40f486:	49 01 ec             	add    %rbp,%r12
  40f489:	e8 f2 2f ff ff       	callq  402480 <memset@plt>
  40f48e:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
  40f493:	8b 4c 24 28          	mov    0x28(%rsp),%ecx
  40f497:	44 8b 4c 24 18       	mov    0x18(%rsp),%r9d
  40f49c:	84 c9                	test   %cl,%cl
  40f49e:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40f4a3:	4c 89 fa             	mov    %r15,%rdx
  40f4a6:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  40f4ab:	4c 89 e7             	mov    %r12,%rdi
  40f4ae:	0f 85 c9 0e 00 00    	jne    41037d <__sprintf_chk@plt+0xdaed>
  40f4b4:	45 84 c9             	test   %r9b,%r9b
  40f4b7:	0f 84 07 0e 00 00    	je     4102c4 <__sprintf_chk@plt+0xda34>
  40f4bd:	e8 be f7 ff ff       	callq  40ec80 <__sprintf_chk@plt+0xc3f0>
  40f4c2:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f4c7:	4d 01 fc             	add    %r15,%r12
  40f4ca:	4d 01 c5             	add    %r8,%r13
  40f4cd:	49 89 d8             	mov    %rbx,%r8
  40f4d0:	e9 9f f8 ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  40f4d5:	83 f9 45             	cmp    $0x45,%ecx
  40f4d8:	0f 84 af 0d 00 00    	je     41028d <__sprintf_chk@plt+0xd9fd>
  40f4de:	83 f9 4f             	cmp    $0x4f,%ecx
  40f4e1:	0f 84 d1 fb ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f4e7:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f4ec:	c7 44 24 18 04 00 00 	movl   $0x4,0x18(%rsp)
  40f4f3:	00 
  40f4f4:	44 8b 40 14          	mov    0x14(%rax),%r8d
  40f4f8:	41 81 f8 94 f8 ff ff 	cmp    $0xfffff894,%r8d
  40f4ff:	41 0f 9c c2          	setl   %r10b
  40f503:	41 81 c0 6c 07 00 00 	add    $0x76c,%r8d
  40f50a:	31 ff                	xor    %edi,%edi
  40f50c:	c6 44 24 28 00       	movb   $0x0,0x28(%rsp)
  40f511:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40f518:	44 89 c0             	mov    %r8d,%eax
  40f51b:	48 8d 8c 24 d7 00 00 	lea    0xd7(%rsp),%rcx
  40f522:	00 
  40f523:	be cd cc cc cc       	mov    $0xcccccccd,%esi
  40f528:	f7 d8                	neg    %eax
  40f52a:	45 84 d2             	test   %r10b,%r10b
  40f52d:	44 0f 45 c0          	cmovne %eax,%r8d
  40f531:	eb 08                	jmp    40f53b <__sprintf_chk@plt+0xccab>
  40f533:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40f538:	4c 89 f9             	mov    %r15,%rcx
  40f53b:	40 f6 c7 01          	test   $0x1,%dil
  40f53f:	74 08                	je     40f549 <__sprintf_chk@plt+0xccb9>
  40f541:	c6 41 ff 3a          	movb   $0x3a,-0x1(%rcx)
  40f545:	48 83 e9 01          	sub    $0x1,%rcx
  40f549:	44 89 c0             	mov    %r8d,%eax
  40f54c:	4c 8d 79 ff          	lea    -0x1(%rcx),%r15
  40f550:	f7 e6                	mul    %esi
  40f552:	c1 ea 03             	shr    $0x3,%edx
  40f555:	8d 04 92             	lea    (%rdx,%rdx,4),%eax
  40f558:	01 c0                	add    %eax,%eax
  40f55a:	41 29 c0             	sub    %eax,%r8d
  40f55d:	41 83 c0 30          	add    $0x30,%r8d
  40f561:	d1 ff                	sar    %edi
  40f563:	44 88 41 ff          	mov    %r8b,-0x1(%rcx)
  40f567:	41 89 d0             	mov    %edx,%r8d
  40f56a:	75 cc                	jne    40f538 <__sprintf_chk@plt+0xcca8>
  40f56c:	85 d2                	test   %edx,%edx
  40f56e:	75 c8                	jne    40f538 <__sprintf_chk@plt+0xcca8>
  40f570:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40f574:	39 e8                	cmp    %ebp,%eax
  40f576:	0f 4c c5             	cmovl  %ebp,%eax
  40f579:	45 84 d2             	test   %r10b,%r10b
  40f57c:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40f580:	0f 85 d2 09 00 00    	jne    40ff58 <__sprintf_chk@plt+0xd6c8>
  40f586:	80 7c 24 28 00       	cmpb   $0x0,0x28(%rsp)
  40f58b:	0f 85 ac 0a 00 00    	jne    41003d <__sprintf_chk@plt+0xd7ad>
  40f591:	41 83 fb 2d          	cmp    $0x2d,%r11d
  40f595:	0f 84 c2 0c 00 00    	je     41025d <__sprintf_chk@plt+0xd9cd>
  40f59b:	31 c0                	xor    %eax,%eax
  40f59d:	c6 44 24 30 00       	movb   $0x0,0x30(%rsp)
  40f5a2:	45 31 d2             	xor    %r10d,%r10d
  40f5a5:	0f 1f 00             	nopl   (%rax)
  40f5a8:	44 8b 44 24 18       	mov    0x18(%rsp),%r8d
  40f5ad:	48 8d b4 24 d7 00 00 	lea    0xd7(%rsp),%rsi
  40f5b4:	00 
  40f5b5:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
  40f5ba:	41 29 c0             	sub    %eax,%r8d
  40f5bd:	4c 89 f8             	mov    %r15,%rax
  40f5c0:	48 29 f0             	sub    %rsi,%rax
  40f5c3:	41 01 c0             	add    %eax,%r8d
  40f5c6:	45 85 c0             	test   %r8d,%r8d
  40f5c9:	0f 8e 79 0a 00 00    	jle    410048 <__sprintf_chk@plt+0xd7b8>
  40f5cf:	41 83 fb 5f          	cmp    $0x5f,%r11d
  40f5d3:	0f 84 29 0b 00 00    	je     410102 <__sprintf_chk@plt+0xd872>
  40f5d9:	48 63 54 24 18       	movslq 0x18(%rsp),%rdx
  40f5de:	4c 89 f0             	mov    %r14,%rax
  40f5e1:	4c 29 e8             	sub    %r13,%rax
  40f5e4:	48 39 c2             	cmp    %rax,%rdx
  40f5e7:	0f 83 ab f8 ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40f5ed:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  40f5f2:	0f 84 93 00 00 00    	je     40f68b <__sprintf_chk@plt+0xcdfb>
  40f5f8:	31 d2                	xor    %edx,%edx
  40f5fa:	85 ed                	test   %ebp,%ebp
  40f5fc:	b9 01 00 00 00       	mov    $0x1,%ecx
  40f601:	0f 49 d5             	cmovns %ebp,%edx
  40f604:	48 63 d2             	movslq %edx,%rdx
  40f607:	48 85 d2             	test   %rdx,%rdx
  40f60a:	48 0f 45 ca          	cmovne %rdx,%rcx
  40f60e:	48 39 c8             	cmp    %rcx,%rax
  40f611:	0f 86 81 f8 ff ff    	jbe    40ee98 <__sprintf_chk@plt+0xc608>
  40f617:	4d 85 e4             	test   %r12,%r12
  40f61a:	74 6c                	je     40f688 <__sprintf_chk@plt+0xcdf8>
  40f61c:	48 83 fa 01          	cmp    $0x1,%rdx
  40f620:	76 5e                	jbe    40f680 <__sprintf_chk@plt+0xcdf0>
  40f622:	8b 54 24 18          	mov    0x18(%rsp),%edx
  40f626:	85 d2                	test   %edx,%edx
  40f628:	75 56                	jne    40f680 <__sprintf_chk@plt+0xcdf0>
  40f62a:	48 63 ed             	movslq %ebp,%rbp
  40f62d:	44 89 54 24 60       	mov    %r10d,0x60(%rsp)
  40f632:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  40f637:	48 83 ed 01          	sub    $0x1,%rbp
  40f63b:	41 83 fb 30          	cmp    $0x30,%r11d
  40f63f:	44 89 44 24 50       	mov    %r8d,0x50(%rsp)
  40f644:	44 89 4c 24 38       	mov    %r9d,0x38(%rsp)
  40f649:	44 89 5c 24 30       	mov    %r11d,0x30(%rsp)
  40f64e:	48 89 ea             	mov    %rbp,%rdx
  40f651:	0f 84 6f 0e 00 00    	je     4104c6 <__sprintf_chk@plt+0xdc36>
  40f657:	4c 89 e7             	mov    %r12,%rdi
  40f65a:	be 20 00 00 00       	mov    $0x20,%esi
  40f65f:	49 01 ec             	add    %rbp,%r12
  40f662:	e8 19 2e ff ff       	callq  402480 <memset@plt>
  40f667:	44 8b 54 24 60       	mov    0x60(%rsp),%r10d
  40f66c:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  40f671:	44 8b 44 24 50       	mov    0x50(%rsp),%r8d
  40f676:	44 8b 4c 24 38       	mov    0x38(%rsp),%r9d
  40f67b:	44 8b 5c 24 30       	mov    0x30(%rsp),%r11d
  40f680:	45 88 14 24          	mov    %r10b,(%r12)
  40f684:	49 83 c4 01          	add    $0x1,%r12
  40f688:	49 01 cd             	add    %rcx,%r13
  40f68b:	4d 85 e4             	test   %r12,%r12
  40f68e:	49 63 e8             	movslq %r8d,%rbp
  40f691:	74 2a                	je     40f6bd <__sprintf_chk@plt+0xce2d>
  40f693:	49 63 e8             	movslq %r8d,%rbp
  40f696:	4c 89 e7             	mov    %r12,%rdi
  40f699:	be 30 00 00 00       	mov    $0x30,%esi
  40f69e:	48 89 ea             	mov    %rbp,%rdx
  40f6a1:	44 89 4c 24 38       	mov    %r9d,0x38(%rsp)
  40f6a6:	44 89 5c 24 30       	mov    %r11d,0x30(%rsp)
  40f6ab:	e8 d0 2d ff ff       	callq  402480 <memset@plt>
  40f6b0:	44 8b 4c 24 38       	mov    0x38(%rsp),%r9d
  40f6b5:	44 8b 5c 24 30       	mov    0x30(%rsp),%r11d
  40f6ba:	49 01 ec             	add    %rbp,%r12
  40f6bd:	49 01 ed             	add    %rbp,%r13
  40f6c0:	49 89 d8             	mov    %rbx,%r8
  40f6c3:	31 c9                	xor    %ecx,%ecx
  40f6c5:	31 ed                	xor    %ebp,%ebp
  40f6c7:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
  40f6cc:	49 89 ca             	mov    %rcx,%r10
  40f6cf:	4c 89 f0             	mov    %r14,%rax
  40f6d2:	4c 29 fb             	sub    %r15,%rbx
  40f6d5:	48 39 cb             	cmp    %rcx,%rbx
  40f6d8:	4c 0f 43 d3          	cmovae %rbx,%r10
  40f6dc:	4c 29 e8             	sub    %r13,%rax
  40f6df:	49 39 c2             	cmp    %rax,%r10
  40f6e2:	0f 83 b0 f7 ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40f6e8:	4d 85 e4             	test   %r12,%r12
  40f6eb:	74 7c                	je     40f769 <__sprintf_chk@plt+0xced9>
  40f6ed:	48 39 cb             	cmp    %rcx,%rbx
  40f6f0:	73 49                	jae    40f73b <__sprintf_chk@plt+0xceab>
  40f6f2:	8b 44 24 18          	mov    0x18(%rsp),%eax
  40f6f6:	85 c0                	test   %eax,%eax
  40f6f8:	75 41                	jne    40f73b <__sprintf_chk@plt+0xceab>
  40f6fa:	48 63 ed             	movslq %ebp,%rbp
  40f6fd:	4c 89 54 24 30       	mov    %r10,0x30(%rsp)
  40f702:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
  40f707:	48 29 dd             	sub    %rbx,%rbp
  40f70a:	41 83 fb 30          	cmp    $0x30,%r11d
  40f70e:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40f713:	48 89 ea             	mov    %rbp,%rdx
  40f716:	0f 84 f9 0a 00 00    	je     410215 <__sprintf_chk@plt+0xd985>
  40f71c:	4c 89 e7             	mov    %r12,%rdi
  40f71f:	be 20 00 00 00       	mov    $0x20,%esi
  40f724:	49 01 ec             	add    %rbp,%r12
  40f727:	e8 54 2d ff ff       	callq  402480 <memset@plt>
  40f72c:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
  40f731:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40f736:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f73b:	45 84 c9             	test   %r9b,%r9b
  40f73e:	4c 89 54 24 28       	mov    %r10,0x28(%rsp)
  40f743:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
  40f748:	48 89 da             	mov    %rbx,%rdx
  40f74b:	4c 89 fe             	mov    %r15,%rsi
  40f74e:	4c 89 e7             	mov    %r12,%rdi
  40f751:	0f 84 21 08 00 00    	je     40ff78 <__sprintf_chk@plt+0xd6e8>
  40f757:	e8 24 f5 ff ff       	callq  40ec80 <__sprintf_chk@plt+0xc3f0>
  40f75c:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40f761:	4c 8b 54 24 28       	mov    0x28(%rsp),%r10
  40f766:	49 01 dc             	add    %rbx,%r12
  40f769:	4d 01 d5             	add    %r10,%r13
  40f76c:	e9 03 f6 ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  40f771:	83 f9 45             	cmp    $0x45,%ecx
  40f774:	0f 84 3e f9 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f77a:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
  40f77f:	41 b8 93 24 49 92    	mov    $0x92492493,%r8d
  40f785:	41 8b 42 18          	mov    0x18(%r10),%eax
  40f789:	8d 78 06             	lea    0x6(%rax),%edi
  40f78c:	89 f8                	mov    %edi,%eax
  40f78e:	41 f7 e8             	imul   %r8d
  40f791:	8d 04 3a             	lea    (%rdx,%rdi,1),%eax
  40f794:	89 fa                	mov    %edi,%edx
  40f796:	c1 fa 1f             	sar    $0x1f,%edx
  40f799:	c1 f8 02             	sar    $0x2,%eax
  40f79c:	29 d0                	sub    %edx,%eax
  40f79e:	8d 14 c5 00 00 00 00 	lea    0x0(,%rax,8),%edx
  40f7a5:	29 c2                	sub    %eax,%edx
  40f7a7:	41 8b 42 1c          	mov    0x1c(%r10),%eax
  40f7ab:	29 fa                	sub    %edi,%edx
  40f7ad:	8d 7c 02 07          	lea    0x7(%rdx,%rax,1),%edi
  40f7b1:	89 f8                	mov    %edi,%eax
  40f7b3:	41 f7 e8             	imul   %r8d
  40f7b6:	44 8d 04 3a          	lea    (%rdx,%rdi,1),%r8d
  40f7ba:	c1 ff 1f             	sar    $0x1f,%edi
  40f7bd:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f7c4:	00 
  40f7c5:	41 c1 f8 02          	sar    $0x2,%r8d
  40f7c9:	41 29 f8             	sub    %edi,%r8d
  40f7cc:	e9 57 fa ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f7d1:	83 f9 45             	cmp    $0x45,%ecx
  40f7d4:	0f 84 de f8 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f7da:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f7df:	ba 93 24 49 92       	mov    $0x92492493,%edx
  40f7e4:	8b 78 1c             	mov    0x1c(%rax),%edi
  40f7e7:	2b 78 18             	sub    0x18(%rax),%edi
  40f7ea:	83 c7 07             	add    $0x7,%edi
  40f7ed:	89 f8                	mov    %edi,%eax
  40f7ef:	f7 ea                	imul   %edx
  40f7f1:	eb c3                	jmp    40f7b6 <__sprintf_chk@plt+0xcf26>
  40f7f3:	48 c7 44 24 18 da 64 	movq   $0x4164da,0x18(%rsp)
  40f7fa:	41 00 
  40f7fc:	8b b4 24 10 05 00 00 	mov    0x510(%rsp),%esi
  40f803:	41 0f b6 c1          	movzbl %r9b,%eax
  40f807:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  40f80c:	44 8b 4c 24 44       	mov    0x44(%rsp),%r9d
  40f811:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40f816:	48 c7 c2 ff ff ff ff 	mov    $0xffffffffffffffff,%rdx
  40f81d:	89 c7                	mov    %eax,%edi
  40f81f:	44 89 5c 24 38       	mov    %r11d,0x38(%rsp)
  40f824:	89 44 24 30          	mov    %eax,0x30(%rsp)
  40f828:	89 34 24             	mov    %esi,(%rsp)
  40f82b:	31 f6                	xor    %esi,%esi
  40f82d:	e8 9e f4 ff ff       	callq  40ecd0 <__sprintf_chk@plt+0xc440>
  40f832:	49 89 c7             	mov    %rax,%r15
  40f835:	31 c0                	xor    %eax,%eax
  40f837:	85 ed                	test   %ebp,%ebp
  40f839:	0f 49 c5             	cmovns %ebp,%eax
  40f83c:	4d 89 f2             	mov    %r14,%r10
  40f83f:	48 98                	cltq   
  40f841:	49 39 c7             	cmp    %rax,%r15
  40f844:	48 89 c6             	mov    %rax,%rsi
  40f847:	49 0f 43 f7          	cmovae %r15,%rsi
  40f84b:	4d 29 ea             	sub    %r13,%r10
  40f84e:	4c 39 d6             	cmp    %r10,%rsi
  40f851:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
  40f856:	0f 83 3c f6 ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40f85c:	4d 85 e4             	test   %r12,%r12
  40f85f:	74 62                	je     40f8c3 <__sprintf_chk@plt+0xd033>
  40f861:	49 39 c7             	cmp    %rax,%r15
  40f864:	44 8b 5c 24 38       	mov    0x38(%rsp),%r11d
  40f869:	73 2d                	jae    40f898 <__sprintf_chk@plt+0xd008>
  40f86b:	48 63 ed             	movslq %ebp,%rbp
  40f86e:	4c 89 54 24 38       	mov    %r10,0x38(%rsp)
  40f873:	4c 29 fd             	sub    %r15,%rbp
  40f876:	41 83 fb 30          	cmp    $0x30,%r11d
  40f87a:	48 89 ea             	mov    %rbp,%rdx
  40f87d:	0f 84 27 0a 00 00    	je     4102aa <__sprintf_chk@plt+0xda1a>
  40f883:	4c 89 e7             	mov    %r12,%rdi
  40f886:	be 20 00 00 00       	mov    $0x20,%esi
  40f88b:	49 01 ec             	add    %rbp,%r12
  40f88e:	e8 ed 2b ff ff       	callq  402480 <memset@plt>
  40f893:	4c 8b 54 24 38       	mov    0x38(%rsp),%r10
  40f898:	8b 84 24 10 05 00 00 	mov    0x510(%rsp),%eax
  40f89f:	44 8b 4c 24 44       	mov    0x44(%rsp),%r9d
  40f8a4:	4c 89 e6             	mov    %r12,%rsi
  40f8a7:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  40f8ac:	48 8b 4c 24 18       	mov    0x18(%rsp),%rcx
  40f8b1:	4c 89 d2             	mov    %r10,%rdx
  40f8b4:	8b 7c 24 30          	mov    0x30(%rsp),%edi
  40f8b8:	4d 01 fc             	add    %r15,%r12
  40f8bb:	89 04 24             	mov    %eax,(%rsp)
  40f8be:	e8 0d f4 ff ff       	callq  40ecd0 <__sprintf_chk@plt+0xc440>
  40f8c3:	4c 03 6c 24 28       	add    0x28(%rsp),%r13
  40f8c8:	49 89 d8             	mov    %rbx,%r8
  40f8cb:	e9 a4 f4 ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  40f8d0:	83 f9 45             	cmp    $0x45,%ecx
  40f8d3:	0f 84 df f7 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f8d9:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f8de:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f8e5:	00 
  40f8e6:	44 8b 00             	mov    (%rax),%r8d
  40f8e9:	e9 3a f9 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f8ee:	48 c7 44 24 18 6f 39 	movq   $0x41396f,0x18(%rsp)
  40f8f5:	41 00 
  40f8f7:	e9 00 ff ff ff       	jmpq   40f7fc <__sprintf_chk@plt+0xcf6c>
  40f8fc:	bf 70 00 00 00       	mov    $0x70,%edi
  40f901:	be 70 00 00 00       	mov    $0x70,%esi
  40f906:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40f90c:	84 c0                	test   %al,%al
  40f90e:	ba 00 00 00 00       	mov    $0x0,%edx
  40f913:	b8 01 00 00 00       	mov    $0x1,%eax
  40f918:	44 0f 45 ca          	cmovne %edx,%r9d
  40f91c:	44 0f 45 f8          	cmovne %eax,%r15d
  40f920:	e9 e7 f5 ff ff       	jmpq   40ef0c <__sprintf_chk@plt+0xc67c>
  40f925:	83 f9 45             	cmp    $0x45,%ecx
  40f928:	0f 84 8a f7 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f92e:	83 fd ff             	cmp    $0xffffffff,%ebp
  40f931:	0f 84 a3 0a 00 00    	je     4103da <__sprintf_chk@plt+0xdb4a>
  40f937:	83 fd 08             	cmp    $0x8,%ebp
  40f93a:	0f 8f 99 0c 00 00    	jg     4105d9 <__sprintf_chk@plt+0xdd49>
  40f940:	44 8b 84 24 10 05 00 	mov    0x510(%rsp),%r8d
  40f947:	00 
  40f948:	89 ef                	mov    %ebp,%edi
  40f94a:	41 ba 67 66 66 66    	mov    $0x66666667,%r10d
  40f950:	44 89 c0             	mov    %r8d,%eax
  40f953:	83 c7 01             	add    $0x1,%edi
  40f956:	41 c1 f8 1f          	sar    $0x1f,%r8d
  40f95a:	41 f7 ea             	imul   %r10d
  40f95d:	c1 fa 02             	sar    $0x2,%edx
  40f960:	44 29 c2             	sub    %r8d,%edx
  40f963:	83 ff 09             	cmp    $0x9,%edi
  40f966:	41 89 d0             	mov    %edx,%r8d
  40f969:	75 e5                	jne    40f950 <__sprintf_chk@plt+0xd0c0>
  40f96b:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
  40f96f:	e9 b4 f8 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f974:	83 f9 45             	cmp    $0x45,%ecx
  40f977:	0f 84 3b f7 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40f97d:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f982:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f989:	00 
  40f98a:	44 8b 40 04          	mov    0x4(%rax),%r8d
  40f98e:	e9 95 f8 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f993:	83 f9 45             	cmp    $0x45,%ecx
  40f996:	0f 84 f1 08 00 00    	je     41028d <__sprintf_chk@plt+0xd9fd>
  40f99c:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40f9a1:	41 b8 1f 85 eb 51    	mov    $0x51eb851f,%r8d
  40f9a7:	41 ba 64 00 00 00    	mov    $0x64,%r10d
  40f9ad:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40f9b4:	00 
  40f9b5:	8b 78 14             	mov    0x14(%rax),%edi
  40f9b8:	89 f8                	mov    %edi,%eax
  40f9ba:	41 f7 e8             	imul   %r8d
  40f9bd:	89 f8                	mov    %edi,%eax
  40f9bf:	c1 f8 1f             	sar    $0x1f,%eax
  40f9c2:	41 89 d0             	mov    %edx,%r8d
  40f9c5:	41 c1 f8 05          	sar    $0x5,%r8d
  40f9c9:	41 29 c0             	sub    %eax,%r8d
  40f9cc:	89 f8                	mov    %edi,%eax
  40f9ce:	45 0f af c2          	imul   %r10d,%r8d
  40f9d2:	44 29 c0             	sub    %r8d,%eax
  40f9d5:	41 89 c0             	mov    %eax,%r8d
  40f9d8:	0f 89 4a f8 ff ff    	jns    40f228 <__sprintf_chk@plt+0xc998>
  40f9de:	f7 d8                	neg    %eax
  40f9e0:	41 83 c0 64          	add    $0x64,%r8d
  40f9e4:	81 ff 93 f8 ff ff    	cmp    $0xfffff893,%edi
  40f9ea:	44 0f 4e c0          	cmovle %eax,%r8d
  40f9ee:	e9 35 f8 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40f9f3:	45 31 ff             	xor    %r15d,%r15d
  40f9f6:	e9 11 ff ff ff       	jmpq   40f90c <__sprintf_chk@plt+0xd07c>
  40f9fb:	31 c0                	xor    %eax,%eax
  40f9fd:	85 ed                	test   %ebp,%ebp
  40f9ff:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40fa05:	0f 49 c5             	cmovns %ebp,%eax
  40fa08:	4c 89 f2             	mov    %r14,%rdx
  40fa0b:	48 98                	cltq   
  40fa0d:	48 85 c0             	test   %rax,%rax
  40fa10:	4c 0f 45 f8          	cmovne %rax,%r15
  40fa14:	4c 29 ea             	sub    %r13,%rdx
  40fa17:	49 39 d7             	cmp    %rdx,%r15
  40fa1a:	0f 83 78 f4 ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40fa20:	4d 85 e4             	test   %r12,%r12
  40fa23:	74 33                	je     40fa58 <__sprintf_chk@plt+0xd1c8>
  40fa25:	48 83 f8 01          	cmp    $0x1,%rax
  40fa29:	76 24                	jbe    40fa4f <__sprintf_chk@plt+0xd1bf>
  40fa2b:	48 63 ed             	movslq %ebp,%rbp
  40fa2e:	48 83 ed 01          	sub    $0x1,%rbp
  40fa32:	41 83 fb 30          	cmp    $0x30,%r11d
  40fa36:	48 89 ea             	mov    %rbp,%rdx
  40fa39:	0f 84 ee 0a 00 00    	je     41052d <__sprintf_chk@plt+0xdc9d>
  40fa3f:	4c 89 e7             	mov    %r12,%rdi
  40fa42:	be 20 00 00 00       	mov    $0x20,%esi
  40fa47:	49 01 ec             	add    %rbp,%r12
  40fa4a:	e8 31 2a ff ff       	callq  402480 <memset@plt>
  40fa4f:	41 c6 04 24 09       	movb   $0x9,(%r12)
  40fa54:	49 83 c4 01          	add    $0x1,%r12
  40fa58:	4d 01 fd             	add    %r15,%r13
  40fa5b:	49 89 d8             	mov    %rbx,%r8
  40fa5e:	e9 11 f3 ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  40fa63:	48 8b 74 24 10       	mov    0x10(%rsp),%rsi
  40fa68:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  40fa6d:	44 89 4c 24 28       	mov    %r9d,0x28(%rsp)
  40fa72:	44 89 5c 24 18       	mov    %r11d,0x18(%rsp)
  40fa77:	4c 8d bc 24 d7 00 00 	lea    0xd7(%rsp),%r15
  40fa7e:	00 
  40fa7f:	48 8b 06             	mov    (%rsi),%rax
  40fa82:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  40fa87:	48 8b 46 08          	mov    0x8(%rsi),%rax
  40fa8b:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  40fa90:	48 8b 46 10          	mov    0x10(%rsi),%rax
  40fa94:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  40fa9b:	00 
  40fa9c:	48 8b 46 18          	mov    0x18(%rsi),%rax
  40faa0:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
  40faa7:	00 
  40faa8:	48 8b 46 20          	mov    0x20(%rsi),%rax
  40faac:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  40fab3:	00 
  40fab4:	48 8b 46 28          	mov    0x28(%rsi),%rax
  40fab8:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
  40fabf:	00 
  40fac0:	48 8b 46 30          	mov    0x30(%rsi),%rax
  40fac4:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  40facb:	00 
  40facc:	e8 cf 2b ff ff       	callq  4026a0 <mktime@plt>
  40fad1:	44 8b 5c 24 18       	mov    0x18(%rsp),%r11d
  40fad6:	49 89 c0             	mov    %rax,%r8
  40fad9:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  40fade:	49 c1 e8 3f          	shr    $0x3f,%r8
  40fae2:	48 89 c1             	mov    %rax,%rcx
  40fae5:	49 ba 67 66 66 66 66 	movabs $0x6666666666666667,%r10
  40faec:	66 66 66 
  40faef:	44 89 c6             	mov    %r8d,%esi
  40faf2:	bf 30 00 00 00       	mov    $0x30,%edi
  40faf7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40fafe:	00 00 
  40fb00:	48 89 c8             	mov    %rcx,%rax
  40fb03:	49 83 ef 01          	sub    $0x1,%r15
  40fb07:	49 f7 ea             	imul   %r10
  40fb0a:	48 89 c8             	mov    %rcx,%rax
  40fb0d:	48 c1 f8 3f          	sar    $0x3f,%rax
  40fb11:	48 c1 fa 02          	sar    $0x2,%rdx
  40fb15:	48 29 c2             	sub    %rax,%rdx
  40fb18:	48 8d 04 92          	lea    (%rdx,%rdx,4),%rax
  40fb1c:	48 01 c0             	add    %rax,%rax
  40fb1f:	48 29 c1             	sub    %rax,%rcx
  40fb22:	48 89 c8             	mov    %rcx,%rax
  40fb25:	48 89 d1             	mov    %rdx,%rcx
  40fb28:	89 fa                	mov    %edi,%edx
  40fb2a:	29 c2                	sub    %eax,%edx
  40fb2c:	83 c0 30             	add    $0x30,%eax
  40fb2f:	40 84 f6             	test   %sil,%sil
  40fb32:	0f 44 d0             	cmove  %eax,%edx
  40fb35:	48 85 c9             	test   %rcx,%rcx
  40fb38:	41 88 17             	mov    %dl,(%r15)
  40fb3b:	75 c3                	jne    40fb00 <__sprintf_chk@plt+0xd270>
  40fb3d:	85 ed                	test   %ebp,%ebp
  40fb3f:	b8 01 00 00 00       	mov    $0x1,%eax
  40fb44:	0f 4f c5             	cmovg  %ebp,%eax
  40fb47:	45 84 c0             	test   %r8b,%r8b
  40fb4a:	89 44 24 18          	mov    %eax,0x18(%rsp)
  40fb4e:	0f 84 3d fa ff ff    	je     40f591 <__sprintf_chk@plt+0xcd01>
  40fb54:	41 83 fb 2d          	cmp    $0x2d,%r11d
  40fb58:	0f 84 2e 04 00 00    	je     40ff8c <__sprintf_chk@plt+0xd6fc>
  40fb5e:	b8 01 00 00 00       	mov    $0x1,%eax
  40fb63:	c6 44 24 30 01       	movb   $0x1,0x30(%rsp)
  40fb68:	41 ba 2d 00 00 00    	mov    $0x2d,%r10d
  40fb6e:	e9 35 fa ff ff       	jmpq   40f5a8 <__sprintf_chk@plt+0xcd18>
  40fb73:	31 c0                	xor    %eax,%eax
  40fb75:	85 ed                	test   %ebp,%ebp
  40fb77:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40fb7d:	0f 49 c5             	cmovns %ebp,%eax
  40fb80:	4c 89 f2             	mov    %r14,%rdx
  40fb83:	48 98                	cltq   
  40fb85:	48 85 c0             	test   %rax,%rax
  40fb88:	4c 0f 45 f8          	cmovne %rax,%r15
  40fb8c:	4c 29 ea             	sub    %r13,%rdx
  40fb8f:	49 39 d7             	cmp    %rdx,%r15
  40fb92:	0f 83 00 f3 ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40fb98:	4d 85 e4             	test   %r12,%r12
  40fb9b:	0f 84 b7 fe ff ff    	je     40fa58 <__sprintf_chk@plt+0xd1c8>
  40fba1:	48 83 f8 01          	cmp    $0x1,%rax
  40fba5:	76 24                	jbe    40fbcb <__sprintf_chk@plt+0xd33b>
  40fba7:	48 63 ed             	movslq %ebp,%rbp
  40fbaa:	48 83 ed 01          	sub    $0x1,%rbp
  40fbae:	41 83 fb 30          	cmp    $0x30,%r11d
  40fbb2:	48 89 ea             	mov    %rbp,%rdx
  40fbb5:	0f 84 5d 09 00 00    	je     410518 <__sprintf_chk@plt+0xdc88>
  40fbbb:	4c 89 e7             	mov    %r12,%rdi
  40fbbe:	be 20 00 00 00       	mov    $0x20,%esi
  40fbc3:	49 01 ec             	add    %rbp,%r12
  40fbc6:	e8 b5 28 ff ff       	callq  402480 <memset@plt>
  40fbcb:	41 c6 04 24 0a       	movb   $0xa,(%r12)
  40fbd0:	49 83 c4 01          	add    $0x1,%r12
  40fbd4:	e9 7f fe ff ff       	jmpq   40fa58 <__sprintf_chk@plt+0xd1c8>
  40fbd9:	83 f9 45             	cmp    $0x45,%ecx
  40fbdc:	0f 84 d6 f4 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40fbe2:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40fbe7:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40fbee:	00 
  40fbef:	44 8b 40 08          	mov    0x8(%rax),%r8d
  40fbf3:	e9 30 f6 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40fbf8:	83 f9 45             	cmp    $0x45,%ecx
  40fbfb:	0f 84 b7 f4 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40fc01:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
  40fc06:	41 8b 47 14          	mov    0x14(%r15),%eax
  40fc0a:	89 c2                	mov    %eax,%edx
  40fc0c:	89 44 24 28          	mov    %eax,0x28(%rsp)
  40fc10:	c1 f8 1f             	sar    $0x1f,%eax
  40fc13:	25 90 01 00 00       	and    $0x190,%eax
  40fc18:	44 8d 54 10 9c       	lea    -0x64(%rax,%rdx,1),%r10d
  40fc1d:	41 8b 57 18          	mov    0x18(%r15),%edx
  40fc21:	45 8b 7f 1c          	mov    0x1c(%r15),%r15d
  40fc25:	45 89 f8             	mov    %r15d,%r8d
  40fc28:	89 54 24 18          	mov    %edx,0x18(%rsp)
  40fc2c:	41 29 d0             	sub    %edx,%r8d
  40fc2f:	ba 93 24 49 92       	mov    $0x92492493,%edx
  40fc34:	41 81 c0 7e 01 00 00 	add    $0x17e,%r8d
  40fc3b:	44 89 c0             	mov    %r8d,%eax
  40fc3e:	f7 ea                	imul   %edx
  40fc40:	42 8d 04 02          	lea    (%rdx,%r8,1),%eax
  40fc44:	44 89 c2             	mov    %r8d,%edx
  40fc47:	c1 fa 1f             	sar    $0x1f,%edx
  40fc4a:	c1 f8 02             	sar    $0x2,%eax
  40fc4d:	29 d0                	sub    %edx,%eax
  40fc4f:	8d 14 c5 00 00 00 00 	lea    0x0(,%rax,8),%edx
  40fc56:	29 c2                	sub    %eax,%edx
  40fc58:	44 89 f8             	mov    %r15d,%eax
  40fc5b:	44 29 c0             	sub    %r8d,%eax
  40fc5e:	44 8d 44 10 03       	lea    0x3(%rax,%rdx,1),%r8d
  40fc63:	45 85 c0             	test   %r8d,%r8d
  40fc66:	0f 88 88 07 00 00    	js     4103f4 <__sprintf_chk@plt+0xdb64>
  40fc6c:	41 f6 c2 03          	test   $0x3,%r10b
  40fc70:	b8 93 fe ff ff       	mov    $0xfffffe93,%eax
  40fc75:	75 4e                	jne    40fcc5 <__sprintf_chk@plt+0xd435>
  40fc77:	44 89 d0             	mov    %r10d,%eax
  40fc7a:	ba 1f 85 eb 51       	mov    $0x51eb851f,%edx
  40fc7f:	f7 ea                	imul   %edx
  40fc81:	44 89 d0             	mov    %r10d,%eax
  40fc84:	c1 f8 1f             	sar    $0x1f,%eax
  40fc87:	89 44 24 38          	mov    %eax,0x38(%rsp)
  40fc8b:	89 54 24 30          	mov    %edx,0x30(%rsp)
  40fc8f:	c1 fa 05             	sar    $0x5,%edx
  40fc92:	29 c2                	sub    %eax,%edx
  40fc94:	b8 64 00 00 00       	mov    $0x64,%eax
  40fc99:	0f af d0             	imul   %eax,%edx
  40fc9c:	b8 92 fe ff ff       	mov    $0xfffffe92,%eax
  40fca1:	41 39 d2             	cmp    %edx,%r10d
  40fca4:	75 1f                	jne    40fcc5 <__sprintf_chk@plt+0xd435>
  40fca6:	8b 44 24 30          	mov    0x30(%rsp),%eax
  40fcaa:	c1 f8 07             	sar    $0x7,%eax
  40fcad:	2b 44 24 38          	sub    0x38(%rsp),%eax
  40fcb1:	69 c0 90 01 00 00    	imul   $0x190,%eax,%eax
  40fcb7:	41 29 c2             	sub    %eax,%r10d
  40fcba:	41 83 fa 01          	cmp    $0x1,%r10d
  40fcbe:	19 c0                	sbb    %eax,%eax
  40fcc0:	2d 6d 01 00 00       	sub    $0x16d,%eax
  40fcc5:	45 8d 14 07          	lea    (%r15,%rax,1),%r10d
  40fcc9:	ba 93 24 49 92       	mov    $0x92492493,%edx
  40fcce:	44 89 d0             	mov    %r10d,%eax
  40fcd1:	2b 44 24 18          	sub    0x18(%rsp),%eax
  40fcd5:	44 8d b8 7e 01 00 00 	lea    0x17e(%rax),%r15d
  40fcdc:	44 89 f8             	mov    %r15d,%eax
  40fcdf:	45 29 fa             	sub    %r15d,%r10d
  40fce2:	f7 ea                	imul   %edx
  40fce4:	42 8d 04 3a          	lea    (%rdx,%r15,1),%eax
  40fce8:	44 89 fa             	mov    %r15d,%edx
  40fceb:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40fcf1:	c1 fa 1f             	sar    $0x1f,%edx
  40fcf4:	c1 f8 02             	sar    $0x2,%eax
  40fcf7:	29 d0                	sub    %edx,%eax
  40fcf9:	8d 14 c5 00 00 00 00 	lea    0x0(,%rax,8),%edx
  40fd00:	29 c2                	sub    %eax,%edx
  40fd02:	45 8d 54 12 03       	lea    0x3(%r10,%rdx,1),%r10d
  40fd07:	45 85 d2             	test   %r10d,%r10d
  40fd0a:	0f 88 43 08 00 00    	js     410553 <__sprintf_chk@plt+0xdcc3>
  40fd10:	40 80 ff 47          	cmp    $0x47,%dil
  40fd14:	0f 84 72 06 00 00    	je     41038c <__sprintf_chk@plt+0xdafc>
  40fd1a:	40 80 ff 67          	cmp    $0x67,%dil
  40fd1e:	0f 85 05 06 00 00    	jne    410329 <__sprintf_chk@plt+0xda99>
  40fd24:	8b 44 24 28          	mov    0x28(%rsp),%eax
  40fd28:	41 b8 1f 85 eb 51    	mov    $0x51eb851f,%r8d
  40fd2e:	8b 7c 24 28          	mov    0x28(%rsp),%edi
  40fd32:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40fd39:	00 
  40fd3a:	41 f7 e8             	imul   %r8d
  40fd3d:	89 f8                	mov    %edi,%eax
  40fd3f:	c1 f8 1f             	sar    $0x1f,%eax
  40fd42:	c1 fa 05             	sar    $0x5,%edx
  40fd45:	29 c2                	sub    %eax,%edx
  40fd47:	b8 64 00 00 00       	mov    $0x64,%eax
  40fd4c:	0f af d0             	imul   %eax,%edx
  40fd4f:	89 f8                	mov    %edi,%eax
  40fd51:	29 d0                	sub    %edx,%eax
  40fd53:	89 c7                	mov    %eax,%edi
  40fd55:	44 01 ff             	add    %r15d,%edi
  40fd58:	89 f8                	mov    %edi,%eax
  40fd5a:	41 f7 e8             	imul   %r8d
  40fd5d:	89 f8                	mov    %edi,%eax
  40fd5f:	c1 f8 1f             	sar    $0x1f,%eax
  40fd62:	41 89 d0             	mov    %edx,%r8d
  40fd65:	41 c1 f8 05          	sar    $0x5,%r8d
  40fd69:	41 29 c0             	sub    %eax,%r8d
  40fd6c:	b8 64 00 00 00       	mov    $0x64,%eax
  40fd71:	44 0f af c0          	imul   %eax,%r8d
  40fd75:	44 29 c7             	sub    %r8d,%edi
  40fd78:	41 89 f8             	mov    %edi,%r8d
  40fd7b:	0f 89 a7 f4 ff ff    	jns    40f228 <__sprintf_chk@plt+0xc998>
  40fd81:	8b 7c 24 28          	mov    0x28(%rsp),%edi
  40fd85:	b8 94 f8 ff ff       	mov    $0xfffff894,%eax
  40fd8a:	44 89 c2             	mov    %r8d,%edx
  40fd8d:	44 29 f8             	sub    %r15d,%eax
  40fd90:	f7 da                	neg    %edx
  40fd92:	41 83 c0 64          	add    $0x64,%r8d
  40fd96:	39 c7                	cmp    %eax,%edi
  40fd98:	44 0f 4c c2          	cmovl  %edx,%r8d
  40fd9c:	e9 87 f4 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40fda1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  40fda8:	85 c9                	test   %ecx,%ecx
  40fdaa:	0f 85 08 f3 ff ff    	jne    40f0b8 <__sprintf_chk@plt+0xc828>
  40fdb0:	48 c7 44 24 18 d1 64 	movq   $0x4164d1,0x18(%rsp)
  40fdb7:	41 00 
  40fdb9:	e9 3e fa ff ff       	jmpq   40f7fc <__sprintf_chk@plt+0xcf6c>
  40fdbe:	85 c9                	test   %ecx,%ecx
  40fdc0:	0f 85 f2 f2 ff ff    	jne    40f0b8 <__sprintf_chk@plt+0xc828>
  40fdc6:	48 c7 44 24 18 c8 64 	movq   $0x4164c8,0x18(%rsp)
  40fdcd:	41 00 
  40fdcf:	e9 28 fa ff ff       	jmpq   40f7fc <__sprintf_chk@plt+0xcf6c>
  40fdd4:	83 f9 4f             	cmp    $0x4f,%ecx
  40fdd7:	0f 84 db f2 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40fddd:	83 f9 45             	cmp    $0x45,%ecx
  40fde0:	0f 84 d4 05 00 00    	je     4103ba <__sprintf_chk@plt+0xdb2a>
  40fde6:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40fdeb:	ba 1f 85 eb 51       	mov    $0x51eb851f,%edx
  40fdf0:	41 ba 64 00 00 00    	mov    $0x64,%r10d
  40fdf6:	8b 48 14             	mov    0x14(%rax),%ecx
  40fdf9:	89 c8                	mov    %ecx,%eax
  40fdfb:	f7 ea                	imul   %edx
  40fdfd:	89 c8                	mov    %ecx,%eax
  40fdff:	c1 f8 1f             	sar    $0x1f,%eax
  40fe02:	c1 fa 05             	sar    $0x5,%edx
  40fe05:	29 c2                	sub    %eax,%edx
  40fe07:	31 c0                	xor    %eax,%eax
  40fe09:	44 8d 42 13          	lea    0x13(%rdx),%r8d
  40fe0d:	41 0f af d2          	imul   %r10d,%edx
  40fe11:	39 d1                	cmp    %edx,%ecx
  40fe13:	0f 88 45 07 00 00    	js     41055e <__sprintf_chk@plt+0xdcce>
  40fe19:	81 f9 94 f8 ff ff    	cmp    $0xfffff894,%ecx
  40fe1f:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40fe26:	00 
  40fe27:	41 0f 9c c2          	setl   %r10b
  40fe2b:	41 29 c0             	sub    %eax,%r8d
  40fe2e:	e9 d7 f6 ff ff       	jmpq   40f50a <__sprintf_chk@plt+0xcc7a>
  40fe33:	0f b6 43 01          	movzbl 0x1(%rbx),%eax
  40fe37:	3c 3a                	cmp    $0x3a,%al
  40fe39:	0f 84 c7 04 00 00    	je     410306 <__sprintf_chk@plt+0xda76>
  40fe3f:	48 8d 53 01          	lea    0x1(%rbx),%rdx
  40fe43:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40fe49:	3c 7a                	cmp    $0x7a,%al
  40fe4b:	0f 85 67 f2 ff ff    	jne    40f0b8 <__sprintf_chk@plt+0xc828>
  40fe51:	48 89 d3             	mov    %rdx,%rbx
  40fe54:	e9 a7 f4 ff ff       	jmpq   40f300 <__sprintf_chk@plt+0xca70>
  40fe59:	83 f9 45             	cmp    $0x45,%ecx
  40fe5c:	0f 84 56 f2 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40fe62:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40fe67:	c6 44 24 28 00       	movb   $0x0,0x28(%rsp)
  40fe6c:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40fe73:	00 
  40fe74:	44 8b 40 10          	mov    0x10(%rax),%r8d
  40fe78:	41 83 f8 ff          	cmp    $0xffffffff,%r8d
  40fe7c:	41 0f 9c c2          	setl   %r10b
  40fe80:	41 83 c0 01          	add    $0x1,%r8d
  40fe84:	31 ff                	xor    %edi,%edi
  40fe86:	e9 ad f3 ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  40fe8b:	85 c9                	test   %ecx,%ecx
  40fe8d:	0f 85 07 04 00 00    	jne    41029a <__sprintf_chk@plt+0xda0a>
  40fe93:	85 ed                	test   %ebp,%ebp
  40fe95:	89 c8                	mov    %ecx,%eax
  40fe97:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  40fe9d:	0f 49 c5             	cmovns %ebp,%eax
  40fea0:	4c 89 f2             	mov    %r14,%rdx
  40fea3:	48 98                	cltq   
  40fea5:	48 85 c0             	test   %rax,%rax
  40fea8:	4c 0f 45 f8          	cmovne %rax,%r15
  40feac:	4c 29 ea             	sub    %r13,%rdx
  40feaf:	49 39 d7             	cmp    %rdx,%r15
  40feb2:	0f 83 e0 ef ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40feb8:	4d 85 e4             	test   %r12,%r12
  40febb:	0f 84 97 fb ff ff    	je     40fa58 <__sprintf_chk@plt+0xd1c8>
  40fec1:	48 83 f8 01          	cmp    $0x1,%rax
  40fec5:	76 27                	jbe    40feee <__sprintf_chk@plt+0xd65e>
  40fec7:	48 63 ed             	movslq %ebp,%rbp
  40feca:	48 83 ed 01          	sub    $0x1,%rbp
  40fece:	41 83 fb 30          	cmp    $0x30,%r11d
  40fed2:	48 89 ea             	mov    %rbp,%rdx
  40fed5:	0f 84 e6 06 00 00    	je     4105c1 <__sprintf_chk@plt+0xdd31>
  40fedb:	4c 89 e7             	mov    %r12,%rdi
  40fede:	be 20 00 00 00       	mov    $0x20,%esi
  40fee3:	49 01 ec             	add    %rbp,%r12
  40fee6:	e8 95 25 ff ff       	callq  402480 <memset@plt>
  40feeb:	0f b6 3b             	movzbl (%rbx),%edi
  40feee:	41 88 3c 24          	mov    %dil,(%r12)
  40fef2:	49 83 c4 01          	add    $0x1,%r12
  40fef6:	e9 5d fb ff ff       	jmpq   40fa58 <__sprintf_chk@plt+0xd1c8>
  40fefb:	4c 8d 43 ff          	lea    -0x1(%rbx),%r8
  40feff:	0f b6 7b ff          	movzbl -0x1(%rbx),%edi
  40ff03:	e9 97 f4 ff ff       	jmpq   40f39f <__sprintf_chk@plt+0xcb0f>
  40ff08:	83 f9 45             	cmp    $0x45,%ecx
  40ff0b:	0f 84 a7 f1 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40ff11:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  40ff16:	44 8b 40 08          	mov    0x8(%rax),%r8d
  40ff1a:	e9 98 f3 ff ff       	jmpq   40f2b7 <__sprintf_chk@plt+0xca27>
  40ff1f:	83 f9 45             	cmp    $0x45,%ecx
  40ff22:	0f 84 90 f1 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40ff28:	44 8b 44 24 40       	mov    0x40(%rsp),%r8d
  40ff2d:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  40ff34:	00 
  40ff35:	e9 ee f2 ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  40ff3a:	83 f9 45             	cmp    $0x45,%ecx
  40ff3d:	0f 84 75 f1 ff ff    	je     40f0b8 <__sprintf_chk@plt+0xc828>
  40ff43:	44 8b 44 24 40       	mov    0x40(%rsp),%r8d
  40ff48:	e9 6a f3 ff ff       	jmpq   40f2b7 <__sprintf_chk@plt+0xca27>
  40ff4d:	49 89 d8             	mov    %rbx,%r8
  40ff50:	e9 4a f4 ff ff       	jmpq   40f39f <__sprintf_chk@plt+0xcb0f>
  40ff55:	0f 1f 00             	nopl   (%rax)
  40ff58:	41 ba 2d 00 00 00    	mov    $0x2d,%r10d
  40ff5e:	41 83 fb 2d          	cmp    $0x2d,%r11d
  40ff62:	74 2e                	je     40ff92 <__sprintf_chk@plt+0xd702>
  40ff64:	b8 01 00 00 00       	mov    $0x1,%eax
  40ff69:	c6 44 24 30 01       	movb   $0x1,0x30(%rsp)
  40ff6e:	e9 35 f6 ff ff       	jmpq   40f5a8 <__sprintf_chk@plt+0xcd18>
  40ff73:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  40ff78:	e8 43 26 ff ff       	callq  4025c0 <memcpy@plt>
  40ff7d:	4c 8b 54 24 28       	mov    0x28(%rsp),%r10
  40ff82:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40ff87:	e9 da f7 ff ff       	jmpq   40f766 <__sprintf_chk@plt+0xced6>
  40ff8c:	41 ba 2d 00 00 00    	mov    $0x2d,%r10d
  40ff92:	31 c9                	xor    %ecx,%ecx
  40ff94:	85 ed                	test   %ebp,%ebp
  40ff96:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  40ff9c:	0f 49 cd             	cmovns %ebp,%ecx
  40ff9f:	4c 89 f0             	mov    %r14,%rax
  40ffa2:	48 63 c9             	movslq %ecx,%rcx
  40ffa5:	48 85 c9             	test   %rcx,%rcx
  40ffa8:	4c 0f 45 c1          	cmovne %rcx,%r8
  40ffac:	4c 29 e8             	sub    %r13,%rax
  40ffaf:	49 39 c0             	cmp    %rax,%r8
  40ffb2:	0f 83 e0 ee ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  40ffb8:	4d 85 e4             	test   %r12,%r12
  40ffbb:	74 62                	je     41001f <__sprintf_chk@plt+0xd78f>
  40ffbd:	48 83 f9 01          	cmp    $0x1,%rcx
  40ffc1:	76 54                	jbe    410017 <__sprintf_chk@plt+0xd787>
  40ffc3:	8b 7c 24 18          	mov    0x18(%rsp),%edi
  40ffc7:	85 ff                	test   %edi,%edi
  40ffc9:	75 4c                	jne    410017 <__sprintf_chk@plt+0xd787>
  40ffcb:	48 63 c5             	movslq %ebp,%rax
  40ffce:	4c 89 e7             	mov    %r12,%rdi
  40ffd1:	be 20 00 00 00       	mov    $0x20,%esi
  40ffd6:	4c 8d 58 ff          	lea    -0x1(%rax),%r11
  40ffda:	44 89 54 24 58       	mov    %r10d,0x58(%rsp)
  40ffdf:	4c 89 44 24 50       	mov    %r8,0x50(%rsp)
  40ffe4:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  40ffe9:	44 89 4c 24 30       	mov    %r9d,0x30(%rsp)
  40ffee:	4c 89 da             	mov    %r11,%rdx
  40fff1:	4c 89 5c 24 28       	mov    %r11,0x28(%rsp)
  40fff6:	e8 85 24 ff ff       	callq  402480 <memset@plt>
  40fffb:	4c 8b 5c 24 28       	mov    0x28(%rsp),%r11
  410000:	44 8b 54 24 58       	mov    0x58(%rsp),%r10d
  410005:	4c 8b 44 24 50       	mov    0x50(%rsp),%r8
  41000a:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
  41000f:	44 8b 4c 24 30       	mov    0x30(%rsp),%r9d
  410014:	4d 01 dc             	add    %r11,%r12
  410017:	45 88 14 24          	mov    %r10b,(%r12)
  41001b:	49 83 c4 01          	add    $0x1,%r12
  41001f:	48 8d 84 24 d7 00 00 	lea    0xd7(%rsp),%rax
  410026:	00 
  410027:	4d 01 c5             	add    %r8,%r13
  41002a:	41 bb 2d 00 00 00    	mov    $0x2d,%r11d
  410030:	49 89 d8             	mov    %rbx,%r8
  410033:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  410038:	e9 8a f6 ff ff       	jmpq   40f6c7 <__sprintf_chk@plt+0xce37>
  41003d:	41 ba 2b 00 00 00    	mov    $0x2b,%r10d
  410043:	e9 16 ff ff ff       	jmpq   40ff5e <__sprintf_chk@plt+0xd6ce>
  410048:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  41004d:	0f 84 17 02 00 00    	je     41026a <__sprintf_chk@plt+0xd9da>
  410053:	45 31 c0             	xor    %r8d,%r8d
  410056:	85 ed                	test   %ebp,%ebp
  410058:	b9 01 00 00 00       	mov    $0x1,%ecx
  41005d:	44 0f 49 c5          	cmovns %ebp,%r8d
  410061:	4c 89 f0             	mov    %r14,%rax
  410064:	4d 63 c0             	movslq %r8d,%r8
  410067:	4d 85 c0             	test   %r8,%r8
  41006a:	49 0f 45 c8          	cmovne %r8,%rcx
  41006e:	4c 29 e8             	sub    %r13,%rax
  410071:	48 39 c1             	cmp    %rax,%rcx
  410074:	0f 83 1e ee ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  41007a:	4d 85 e4             	test   %r12,%r12
  41007d:	74 75                	je     4100f4 <__sprintf_chk@plt+0xd864>
  41007f:	49 83 f8 01          	cmp    $0x1,%r8
  410083:	76 67                	jbe    4100ec <__sprintf_chk@plt+0xd85c>
  410085:	8b 44 24 18          	mov    0x18(%rsp),%eax
  410089:	85 c0                	test   %eax,%eax
  41008b:	75 5f                	jne    4100ec <__sprintf_chk@plt+0xd85c>
  41008d:	48 63 c5             	movslq %ebp,%rax
  410090:	44 89 54 24 68       	mov    %r10d,0x68(%rsp)
  410095:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  41009a:	48 83 e8 01          	sub    $0x1,%rax
  41009e:	41 83 fb 30          	cmp    $0x30,%r11d
  4100a2:	4c 89 44 24 58       	mov    %r8,0x58(%rsp)
  4100a7:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  4100ac:	44 89 4c 24 50       	mov    %r9d,0x50(%rsp)
  4100b1:	44 89 5c 24 38       	mov    %r11d,0x38(%rsp)
  4100b6:	0f 84 d2 04 00 00    	je     41058e <__sprintf_chk@plt+0xdcfe>
  4100bc:	48 8b 54 24 30       	mov    0x30(%rsp),%rdx
  4100c1:	4c 89 e7             	mov    %r12,%rdi
  4100c4:	be 20 00 00 00       	mov    $0x20,%esi
  4100c9:	e8 b2 23 ff ff       	callq  402480 <memset@plt>
  4100ce:	4c 03 64 24 30       	add    0x30(%rsp),%r12
  4100d3:	44 8b 54 24 68       	mov    0x68(%rsp),%r10d
  4100d8:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  4100dd:	4c 8b 44 24 58       	mov    0x58(%rsp),%r8
  4100e2:	44 8b 4c 24 50       	mov    0x50(%rsp),%r9d
  4100e7:	44 8b 5c 24 38       	mov    0x38(%rsp),%r11d
  4100ec:	45 88 14 24          	mov    %r10b,(%r12)
  4100f0:	49 83 c4 01          	add    $0x1,%r12
  4100f4:	49 01 cd             	add    %rcx,%r13
  4100f7:	4c 89 c1             	mov    %r8,%rcx
  4100fa:	49 89 d8             	mov    %rbx,%r8
  4100fd:	e9 c5 f5 ff ff       	jmpq   40f6c7 <__sprintf_chk@plt+0xce37>
  410102:	4c 89 f0             	mov    %r14,%rax
  410105:	49 63 c8             	movslq %r8d,%rcx
  410108:	4c 29 e8             	sub    %r13,%rax
  41010b:	48 39 c1             	cmp    %rax,%rcx
  41010e:	0f 83 84 ed ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  410114:	4d 85 e4             	test   %r12,%r12
  410117:	74 45                	je     41015e <__sprintf_chk@plt+0xd8ce>
  410119:	48 89 ca             	mov    %rcx,%rdx
  41011c:	4c 89 e7             	mov    %r12,%rdi
  41011f:	be 20 00 00 00       	mov    $0x20,%esi
  410124:	44 89 54 24 68       	mov    %r10d,0x68(%rsp)
  410129:	44 89 44 24 60       	mov    %r8d,0x60(%rsp)
  41012e:	44 89 4c 24 58       	mov    %r9d,0x58(%rsp)
  410133:	44 89 5c 24 50       	mov    %r11d,0x50(%rsp)
  410138:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
  41013d:	e8 3e 23 ff ff       	callq  402480 <memset@plt>
  410142:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
  410147:	44 8b 54 24 68       	mov    0x68(%rsp),%r10d
  41014c:	44 8b 44 24 60       	mov    0x60(%rsp),%r8d
  410151:	44 8b 4c 24 58       	mov    0x58(%rsp),%r9d
  410156:	44 8b 5c 24 50       	mov    0x50(%rsp),%r11d
  41015b:	49 01 cc             	add    %rcx,%r12
  41015e:	49 01 cd             	add    %rcx,%r13
  410161:	41 39 e8             	cmp    %ebp,%r8d
  410164:	0f 8d 1a 01 00 00    	jge    410284 <__sprintf_chk@plt+0xd9f4>
  41016a:	31 c9                	xor    %ecx,%ecx
  41016c:	44 29 c5             	sub    %r8d,%ebp
  41016f:	0f 49 cd             	cmovns %ebp,%ecx
  410172:	48 63 c9             	movslq %ecx,%rcx
  410175:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
  41017a:	0f 84 fc 00 00 00    	je     41027c <__sprintf_chk@plt+0xd9ec>
  410180:	48 85 c9             	test   %rcx,%rcx
  410183:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  410189:	4c 89 f0             	mov    %r14,%rax
  41018c:	4c 0f 45 c1          	cmovne %rcx,%r8
  410190:	4c 29 e8             	sub    %r13,%rax
  410193:	49 39 c0             	cmp    %rax,%r8
  410196:	0f 83 fc ec ff ff    	jae    40ee98 <__sprintf_chk@plt+0xc608>
  41019c:	4d 85 e4             	test   %r12,%r12
  41019f:	74 69                	je     41020a <__sprintf_chk@plt+0xd97a>
  4101a1:	48 83 f9 01          	cmp    $0x1,%rcx
  4101a5:	76 5b                	jbe    410202 <__sprintf_chk@plt+0xd972>
  4101a7:	8b 74 24 18          	mov    0x18(%rsp),%esi
  4101ab:	85 f6                	test   %esi,%esi
  4101ad:	75 53                	jne    410202 <__sprintf_chk@plt+0xd972>
  4101af:	48 63 c5             	movslq %ebp,%rax
  4101b2:	4c 89 e7             	mov    %r12,%rdi
  4101b5:	be 20 00 00 00       	mov    $0x20,%esi
  4101ba:	48 83 e8 01          	sub    $0x1,%rax
  4101be:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  4101c3:	44 89 54 24 60       	mov    %r10d,0x60(%rsp)
  4101c8:	48 89 c2             	mov    %rax,%rdx
  4101cb:	4c 89 44 24 58       	mov    %r8,0x58(%rsp)
  4101d0:	44 89 4c 24 50       	mov    %r9d,0x50(%rsp)
  4101d5:	44 89 5c 24 38       	mov    %r11d,0x38(%rsp)
  4101da:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  4101df:	e8 9c 22 ff ff       	callq  402480 <memset@plt>
  4101e4:	4c 03 64 24 30       	add    0x30(%rsp),%r12
  4101e9:	48 8b 4c 24 68       	mov    0x68(%rsp),%rcx
  4101ee:	44 8b 54 24 60       	mov    0x60(%rsp),%r10d
  4101f3:	4c 8b 44 24 58       	mov    0x58(%rsp),%r8
  4101f8:	44 8b 4c 24 50       	mov    0x50(%rsp),%r9d
  4101fd:	44 8b 5c 24 38       	mov    0x38(%rsp),%r11d
  410202:	45 88 14 24          	mov    %r10b,(%r12)
  410206:	49 83 c4 01          	add    $0x1,%r12
  41020a:	4d 01 c5             	add    %r8,%r13
  41020d:	49 89 d8             	mov    %rbx,%r8
  410210:	e9 b2 f4 ff ff       	jmpq   40f6c7 <__sprintf_chk@plt+0xce37>
  410215:	4c 89 e7             	mov    %r12,%rdi
  410218:	be 30 00 00 00       	mov    $0x30,%esi
  41021d:	49 01 ec             	add    %rbp,%r12
  410220:	e8 5b 22 ff ff       	callq  402480 <memset@plt>
  410225:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  41022a:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  41022f:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
  410234:	e9 02 f5 ff ff       	jmpq   40f73b <__sprintf_chk@plt+0xceab>
  410239:	4c 89 e7             	mov    %r12,%rdi
  41023c:	be 30 00 00 00       	mov    $0x30,%esi
  410241:	49 01 ec             	add    %rbp,%r12
  410244:	e8 37 22 ff ff       	callq  402480 <memset@plt>
  410249:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  41024e:	44 8b 4c 24 28       	mov    0x28(%rsp),%r9d
  410253:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
  410258:	e9 a4 ed ff ff       	jmpq   40f001 <__sprintf_chk@plt+0xc771>
  41025d:	48 8d 84 24 d7 00 00 	lea    0xd7(%rsp),%rax
  410264:	00 
  410265:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  41026a:	31 c9                	xor    %ecx,%ecx
  41026c:	85 ed                	test   %ebp,%ebp
  41026e:	49 89 d8             	mov    %rbx,%r8
  410271:	0f 49 cd             	cmovns %ebp,%ecx
  410274:	48 63 c9             	movslq %ecx,%rcx
  410277:	e9 4b f4 ff ff       	jmpq   40f6c7 <__sprintf_chk@plt+0xce37>
  41027c:	49 89 d8             	mov    %rbx,%r8
  41027f:	e9 43 f4 ff ff       	jmpq   40f6c7 <__sprintf_chk@plt+0xce37>
  410284:	31 c9                	xor    %ecx,%ecx
  410286:	31 ed                	xor    %ebp,%ebp
  410288:	e9 e8 fe ff ff       	jmpq   410175 <__sprintf_chk@plt+0xd8e5>
  41028d:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
  410294:	00 
  410295:	e9 b0 ef ff ff       	jmpq   40f24a <__sprintf_chk@plt+0xc9ba>
  41029a:	49 89 d8             	mov    %rbx,%r8
  41029d:	e9 07 f1 ff ff       	jmpq   40f3a9 <__sprintf_chk@plt+0xcb19>
  4102a2:	49 89 d8             	mov    %rbx,%r8
  4102a5:	e9 ca ea ff ff       	jmpq   40ed74 <__sprintf_chk@plt+0xc4e4>
  4102aa:	4c 89 e7             	mov    %r12,%rdi
  4102ad:	be 30 00 00 00       	mov    $0x30,%esi
  4102b2:	49 01 ec             	add    %rbp,%r12
  4102b5:	e8 c6 21 ff ff       	callq  402480 <memset@plt>
  4102ba:	4c 8b 54 24 38       	mov    0x38(%rsp),%r10
  4102bf:	e9 d4 f5 ff ff       	jmpq   40f898 <__sprintf_chk@plt+0xd008>
  4102c4:	e8 f7 22 ff ff       	callq  4025c0 <memcpy@plt>
  4102c9:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  4102ce:	e9 f4 f1 ff ff       	jmpq   40f4c7 <__sprintf_chk@plt+0xcc37>
  4102d3:	41 69 f8 10 27 00 00 	imul   $0x2710,%r8d,%edi
  4102da:	41 b8 64 00 00 00    	mov    $0x64,%r8d
  4102e0:	41 c1 ea 1f          	shr    $0x1f,%r10d
  4102e4:	41 0f af c0          	imul   %r8d,%eax
  4102e8:	c6 44 24 28 01       	movb   $0x1,0x28(%rsp)
  4102ed:	c7 44 24 18 09 00 00 	movl   $0x9,0x18(%rsp)
  4102f4:	00 
  4102f5:	44 8d 04 07          	lea    (%rdi,%rax,1),%r8d
  4102f9:	bf 14 00 00 00       	mov    $0x14,%edi
  4102fe:	41 01 d0             	add    %edx,%r8d
  410301:	e9 32 ef ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  410306:	48 8d 7b 02          	lea    0x2(%rbx),%rdi
  41030a:	41 bf 01 00 00 00    	mov    $0x1,%r15d
  410310:	48 89 fa             	mov    %rdi,%rdx
  410313:	48 83 c7 01          	add    $0x1,%rdi
  410317:	0f b6 47 ff          	movzbl -0x1(%rdi),%eax
  41031b:	49 83 c7 01          	add    $0x1,%r15
  41031f:	3c 3a                	cmp    $0x3a,%al
  410321:	0f 85 22 fb ff ff    	jne    40fe49 <__sprintf_chk@plt+0xd5b9>
  410327:	eb e7                	jmp    410310 <__sprintf_chk@plt+0xda80>
  410329:	44 89 d0             	mov    %r10d,%eax
  41032c:	ba 93 24 49 92       	mov    $0x92492493,%edx
  410331:	c7 44 24 18 02 00 00 	movl   $0x2,0x18(%rsp)
  410338:	00 
  410339:	f7 ea                	imul   %edx
  41033b:	44 89 d0             	mov    %r10d,%eax
  41033e:	c1 f8 1f             	sar    $0x1f,%eax
  410341:	46 8d 04 12          	lea    (%rdx,%r10,1),%r8d
  410345:	41 c1 f8 02          	sar    $0x2,%r8d
  410349:	41 29 c0             	sub    %eax,%r8d
  41034c:	41 83 c0 01          	add    $0x1,%r8d
  410350:	e9 d3 ee ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  410355:	41 bf 64 00 00 00    	mov    $0x64,%r15d
  41035b:	41 c1 ea 1f          	shr    $0x1f,%r10d
  41035f:	bf 04 00 00 00       	mov    $0x4,%edi
  410364:	45 0f af c7          	imul   %r15d,%r8d
  410368:	c6 44 24 28 01       	movb   $0x1,0x28(%rsp)
  41036d:	c7 44 24 18 06 00 00 	movl   $0x6,0x18(%rsp)
  410374:	00 
  410375:	41 01 c0             	add    %eax,%r8d
  410378:	e9 bb ee ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  41037d:	e8 ae e8 ff ff       	callq  40ec30 <__sprintf_chk@plt+0xc3a0>
  410382:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  410387:	e9 3b f1 ff ff       	jmpq   40f4c7 <__sprintf_chk@plt+0xcc37>
  41038c:	8b 7c 24 28          	mov    0x28(%rsp),%edi
  410390:	b8 94 f8 ff ff       	mov    $0xfffff894,%eax
  410395:	c6 44 24 28 00       	movb   $0x0,0x28(%rsp)
  41039a:	44 29 f8             	sub    %r15d,%eax
  41039d:	c7 44 24 18 04 00 00 	movl   $0x4,0x18(%rsp)
  4103a4:	00 
  4103a5:	39 c7                	cmp    %eax,%edi
  4103a7:	45 8d 84 3f 6c 07 00 	lea    0x76c(%r15,%rdi,1),%r8d
  4103ae:	00 
  4103af:	41 0f 9c c2          	setl   %r10b
  4103b3:	31 ff                	xor    %edi,%edi
  4103b5:	e9 7e ee ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  4103ba:	c6 84 24 b0 00 00 00 	movb   $0x20,0xb0(%rsp)
  4103c1:	20 
  4103c2:	c6 84 24 b1 00 00 00 	movb   $0x25,0xb1(%rsp)
  4103c9:	25 
  4103ca:	45 31 ff             	xor    %r15d,%r15d
  4103cd:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
  4103d4:	00 
  4103d5:	e9 83 ee ff ff       	jmpq   40f25d <__sprintf_chk@plt+0xc9cd>
  4103da:	44 8b 84 24 10 05 00 	mov    0x510(%rsp),%r8d
  4103e1:	00 
  4103e2:	bd 09 00 00 00       	mov    $0x9,%ebp
  4103e7:	c7 44 24 18 09 00 00 	movl   $0x9,0x18(%rsp)
  4103ee:	00 
  4103ef:	e9 34 ee ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  4103f4:	41 83 ea 01          	sub    $0x1,%r10d
  4103f8:	b8 6d 01 00 00       	mov    $0x16d,%eax
  4103fd:	41 f6 c2 03          	test   $0x3,%r10b
  410401:	75 50                	jne    410453 <__sprintf_chk@plt+0xdbc3>
  410403:	41 b8 1f 85 eb 51    	mov    $0x51eb851f,%r8d
  410409:	44 89 d0             	mov    %r10d,%eax
  41040c:	41 f7 e8             	imul   %r8d
  41040f:	44 89 d0             	mov    %r10d,%eax
  410412:	c1 f8 1f             	sar    $0x1f,%eax
  410415:	89 44 24 30          	mov    %eax,0x30(%rsp)
  410419:	41 89 d0             	mov    %edx,%r8d
  41041c:	c1 fa 05             	sar    $0x5,%edx
  41041f:	29 c2                	sub    %eax,%edx
  410421:	b8 64 00 00 00       	mov    $0x64,%eax
  410426:	0f af d0             	imul   %eax,%edx
  410429:	b8 6e 01 00 00       	mov    $0x16e,%eax
  41042e:	41 39 d2             	cmp    %edx,%r10d
  410431:	75 20                	jne    410453 <__sprintf_chk@plt+0xdbc3>
  410433:	41 c1 f8 07          	sar    $0x7,%r8d
  410437:	44 2b 44 24 30       	sub    0x30(%rsp),%r8d
  41043c:	45 69 c0 90 01 00 00 	imul   $0x190,%r8d,%r8d
  410443:	45 29 c2             	sub    %r8d,%r10d
  410446:	41 83 fa 01          	cmp    $0x1,%r10d
  41044a:	19 c0                	sbb    %eax,%eax
  41044c:	f7 d0                	not    %eax
  41044e:	05 6e 01 00 00       	add    $0x16e,%eax
  410453:	45 8d 14 07          	lea    (%r15,%rax,1),%r10d
  410457:	ba 93 24 49 92       	mov    $0x92492493,%edx
  41045c:	41 bf ff ff ff ff    	mov    $0xffffffff,%r15d
  410462:	45 89 d0             	mov    %r10d,%r8d
  410465:	44 2b 44 24 18       	sub    0x18(%rsp),%r8d
  41046a:	41 81 c0 7e 01 00 00 	add    $0x17e,%r8d
  410471:	44 89 c0             	mov    %r8d,%eax
  410474:	f7 ea                	imul   %edx
  410476:	42 8d 04 02          	lea    (%rdx,%r8,1),%eax
  41047a:	44 89 c2             	mov    %r8d,%edx
  41047d:	c1 fa 1f             	sar    $0x1f,%edx
  410480:	c1 f8 02             	sar    $0x2,%eax
  410483:	29 d0                	sub    %edx,%eax
  410485:	8d 14 c5 00 00 00 00 	lea    0x0(,%rax,8),%edx
  41048c:	29 c2                	sub    %eax,%edx
  41048e:	44 89 d0             	mov    %r10d,%eax
  410491:	44 29 c0             	sub    %r8d,%eax
  410494:	44 8d 54 10 03       	lea    0x3(%rax,%rdx,1),%r10d
  410499:	e9 72 f8 ff ff       	jmpq   40fd10 <__sprintf_chk@plt+0xd480>
  41049e:	85 d2                	test   %edx,%edx
  4104a0:	0f 85 2d fe ff ff    	jne    4102d3 <__sprintf_chk@plt+0xda43>
  4104a6:	85 c0                	test   %eax,%eax
  4104a8:	0f 85 a7 fe ff ff    	jne    410355 <__sprintf_chk@plt+0xdac5>
  4104ae:	41 c1 ea 1f          	shr    $0x1f,%r10d
  4104b2:	31 ff                	xor    %edi,%edi
  4104b4:	c6 44 24 28 01       	movb   $0x1,0x28(%rsp)
  4104b9:	c7 44 24 18 03 00 00 	movl   $0x3,0x18(%rsp)
  4104c0:	00 
  4104c1:	e9 72 ed ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  4104c6:	4c 89 e7             	mov    %r12,%rdi
  4104c9:	be 30 00 00 00       	mov    $0x30,%esi
  4104ce:	49 01 ec             	add    %rbp,%r12
  4104d1:	e8 aa 1f ff ff       	callq  402480 <memset@plt>
  4104d6:	44 8b 5c 24 30       	mov    0x30(%rsp),%r11d
  4104db:	44 8b 4c 24 38       	mov    0x38(%rsp),%r9d
  4104e0:	44 8b 44 24 50       	mov    0x50(%rsp),%r8d
  4104e5:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  4104ea:	44 8b 54 24 60       	mov    0x60(%rsp),%r10d
  4104ef:	e9 8c f1 ff ff       	jmpq   40f680 <__sprintf_chk@plt+0xcdf0>
  4104f4:	ba 64 00 00 00       	mov    $0x64,%edx
  4104f9:	41 c1 ea 1f          	shr    $0x1f,%r10d
  4104fd:	31 ff                	xor    %edi,%edi
  4104ff:	44 0f af c2          	imul   %edx,%r8d
  410503:	c6 44 24 28 01       	movb   $0x1,0x28(%rsp)
  410508:	c7 44 24 18 05 00 00 	movl   $0x5,0x18(%rsp)
  41050f:	00 
  410510:	41 01 c0             	add    %eax,%r8d
  410513:	e9 20 ed ff ff       	jmpq   40f238 <__sprintf_chk@plt+0xc9a8>
  410518:	4c 89 e7             	mov    %r12,%rdi
  41051b:	be 30 00 00 00       	mov    $0x30,%esi
  410520:	49 01 ec             	add    %rbp,%r12
  410523:	e8 58 1f ff ff       	callq  402480 <memset@plt>
  410528:	e9 9e f6 ff ff       	jmpq   40fbcb <__sprintf_chk@plt+0xd33b>
  41052d:	4c 89 e7             	mov    %r12,%rdi
  410530:	be 30 00 00 00       	mov    $0x30,%esi
  410535:	49 01 ec             	add    %rbp,%r12
  410538:	e8 43 1f ff ff       	callq  402480 <memset@plt>
  41053d:	e9 0d f5 ff ff       	jmpq   40fa4f <__sprintf_chk@plt+0xd1bf>
  410542:	45 31 ff             	xor    %r15d,%r15d
  410545:	48 c7 44 24 48 19 69 	movq   $0x416919,0x48(%rsp)
  41054c:	41 00 
  41054e:	e9 de ee ff ff       	jmpq   40f431 <__sprintf_chk@plt+0xcba1>
  410553:	45 89 c2             	mov    %r8d,%r10d
  410556:	45 30 ff             	xor    %r15b,%r15b
  410559:	e9 b2 f7 ff ff       	jmpq   40fd10 <__sprintf_chk@plt+0xd480>
  41055e:	31 c0                	xor    %eax,%eax
  410560:	45 85 c0             	test   %r8d,%r8d
  410563:	0f 9f c0             	setg   %al
  410566:	e9 ae f8 ff ff       	jmpq   40fe19 <__sprintf_chk@plt+0xd589>
  41056b:	4c 89 e7             	mov    %r12,%rdi
  41056e:	be 30 00 00 00       	mov    $0x30,%esi
  410573:	49 01 ec             	add    %rbp,%r12
  410576:	e8 05 1f ff ff       	callq  402480 <memset@plt>
  41057b:	44 8b 4c 24 18       	mov    0x18(%rsp),%r9d
  410580:	8b 4c 24 28          	mov    0x28(%rsp),%ecx
  410584:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
  410589:	e9 0e ef ff ff       	jmpq   40f49c <__sprintf_chk@plt+0xcc0c>
  41058e:	4c 89 e7             	mov    %r12,%rdi
  410591:	48 89 c2             	mov    %rax,%rdx
  410594:	be 30 00 00 00       	mov    $0x30,%esi
  410599:	e8 e2 1e ff ff       	callq  402480 <memset@plt>
  41059e:	4c 03 64 24 30       	add    0x30(%rsp),%r12
  4105a3:	44 8b 5c 24 38       	mov    0x38(%rsp),%r11d
  4105a8:	44 8b 4c 24 50       	mov    0x50(%rsp),%r9d
  4105ad:	4c 8b 44 24 58       	mov    0x58(%rsp),%r8
  4105b2:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  4105b7:	44 8b 54 24 68       	mov    0x68(%rsp),%r10d
  4105bc:	e9 2b fb ff ff       	jmpq   4100ec <__sprintf_chk@plt+0xd85c>
  4105c1:	4c 89 e7             	mov    %r12,%rdi
  4105c4:	be 30 00 00 00       	mov    $0x30,%esi
  4105c9:	49 01 ec             	add    %rbp,%r12
  4105cc:	e8 af 1e ff ff       	callq  402480 <memset@plt>
  4105d1:	0f b6 3b             	movzbl (%rbx),%edi
  4105d4:	e9 15 f9 ff ff       	jmpq   40feee <__sprintf_chk@plt+0xd65e>
  4105d9:	44 8b 84 24 10 05 00 	mov    0x510(%rsp),%r8d
  4105e0:	00 
  4105e1:	89 6c 24 18          	mov    %ebp,0x18(%rsp)
  4105e5:	e9 3e ec ff ff       	jmpq   40f228 <__sprintf_chk@plt+0xc998>
  4105ea:	e8 b1 1d ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  4105ef:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
  4105f6:	00 
  4105f7:	e9 61 ec ff ff       	jmpq   40f25d <__sprintf_chk@plt+0xc9cd>
  4105fc:	0f 1f 40 00          	nopl   0x0(%rax)
  410600:	48 83 ec 18          	sub    $0x18,%rsp
  410604:	44 89 0c 24          	mov    %r9d,(%rsp)
  410608:	45 89 c1             	mov    %r8d,%r9d
  41060b:	49 89 c8             	mov    %rcx,%r8
  41060e:	48 89 d1             	mov    %rdx,%rcx
  410611:	48 89 f2             	mov    %rsi,%rdx
  410614:	48 89 fe             	mov    %rdi,%rsi
  410617:	31 ff                	xor    %edi,%edi
  410619:	e8 b2 e6 ff ff       	callq  40ecd0 <__sprintf_chk@plt+0xc440>
  41061e:	48 83 c4 18          	add    $0x18,%rsp
  410622:	c3                   	retq   
  410623:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  41062a:	00 00 00 
  41062d:	0f 1f 00             	nopl   (%rax)
  410630:	41 57                	push   %r15
  410632:	41 56                	push   %r14
  410634:	41 55                	push   %r13
  410636:	41 54                	push   %r12
  410638:	4d 89 cc             	mov    %r9,%r12
  41063b:	55                   	push   %rbp
  41063c:	48 89 fd             	mov    %rdi,%rbp
  41063f:	53                   	push   %rbx
  410640:	4c 89 c3             	mov    %r8,%rbx
  410643:	48 83 ec 58          	sub    $0x58,%rsp
  410647:	48 85 f6             	test   %rsi,%rsi
  41064a:	0f 84 e0 03 00 00    	je     410a30 <__sprintf_chk@plt+0xe1a0>
  410650:	49 89 c9             	mov    %rcx,%r9
  410653:	49 89 d0             	mov    %rdx,%r8
  410656:	48 89 f1             	mov    %rsi,%rcx
  410659:	ba c0 68 41 00       	mov    $0x4168c0,%edx
  41065e:	be 01 00 00 00       	mov    $0x1,%esi
  410663:	31 c0                	xor    %eax,%eax
  410665:	e8 a6 21 ff ff       	callq  402810 <__fprintf_chk@plt>
  41066a:	31 ff                	xor    %edi,%edi
  41066c:	ba 05 00 00 00       	mov    $0x5,%edx
  410671:	be d3 68 41 00       	mov    $0x4168d3,%esi
  410676:	e8 e5 1c ff ff       	callq  402360 <dcgettext@plt>
  41067b:	41 b8 dd 07 00 00    	mov    $0x7dd,%r8d
  410681:	48 89 c1             	mov    %rax,%rcx
  410684:	ba a0 6b 41 00       	mov    $0x416ba0,%edx
  410689:	be 01 00 00 00       	mov    $0x1,%esi
  41068e:	48 89 ef             	mov    %rbp,%rdi
  410691:	31 c0                	xor    %eax,%eax
  410693:	e8 78 21 ff ff       	callq  402810 <__fprintf_chk@plt>
  410698:	31 ff                	xor    %edi,%edi
  41069a:	ba 05 00 00 00       	mov    $0x5,%edx
  41069f:	be 30 69 41 00       	mov    $0x416930,%esi
  4106a4:	e8 b7 1c ff ff       	callq  402360 <dcgettext@plt>
  4106a9:	48 89 ee             	mov    %rbp,%rsi
  4106ac:	48 89 c7             	mov    %rax,%rdi
  4106af:	e8 6c 1e ff ff       	callq  402520 <fputs_unlocked@plt>
  4106b4:	49 83 fc 09          	cmp    $0x9,%r12
  4106b8:	0f 87 92 03 00 00    	ja     410a50 <__sprintf_chk@plt+0xe1c0>
  4106be:	42 ff 24 e5 48 6b 41 	jmpq   *0x416b48(,%r12,8)
  4106c5:	00 
  4106c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4106cd:	00 00 00 
  4106d0:	4c 8b 4b 38          	mov    0x38(%rbx),%r9
  4106d4:	4c 8b 43 08          	mov    0x8(%rbx),%r8
  4106d8:	ba 05 00 00 00       	mov    $0x5,%edx
  4106dd:	48 8b 43 10          	mov    0x10(%rbx),%rax
  4106e1:	be a0 6a 41 00       	mov    $0x416aa0,%esi
  4106e6:	31 ff                	xor    %edi,%edi
  4106e8:	4c 8b 6b 30          	mov    0x30(%rbx),%r13
  4106ec:	4c 8b 63 28          	mov    0x28(%rbx),%r12
  4106f0:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  4106f4:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  4106f8:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
  4106fd:	4c 89 44 24 38       	mov    %r8,0x38(%rsp)
  410702:	48 8b 1b             	mov    (%rbx),%rbx
  410705:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  41070a:	e8 51 1c ff ff       	callq  402360 <dcgettext@plt>
  41070f:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
  410714:	4c 8b 44 24 38       	mov    0x38(%rsp),%r8
  410719:	48 89 c2             	mov    %rax,%rdx
  41071c:	4c 89 6c 24 18       	mov    %r13,0x18(%rsp)
  410721:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  410726:	48 89 d9             	mov    %rbx,%rcx
  410729:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  41072e:	4c 89 34 24          	mov    %r14,(%rsp)
  410732:	be 01 00 00 00       	mov    $0x1,%esi
  410737:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  41073c:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
  410741:	48 89 ef             	mov    %rbp,%rdi
  410744:	31 c0                	xor    %eax,%eax
  410746:	e8 c5 20 ff ff       	callq  402810 <__fprintf_chk@plt>
  41074b:	48 83 c4 58          	add    $0x58,%rsp
  41074f:	5b                   	pop    %rbx
  410750:	5d                   	pop    %rbp
  410751:	41 5c                	pop    %r12
  410753:	41 5d                	pop    %r13
  410755:	41 5e                	pop    %r14
  410757:	41 5f                	pop    %r15
  410759:	c3                   	retq   
  41075a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  410760:	4c 8b 53 40          	mov    0x40(%rbx),%r10
  410764:	4c 8b 4b 38          	mov    0x38(%rbx),%r9
  410768:	ba 05 00 00 00       	mov    $0x5,%edx
  41076d:	48 8b 43 10          	mov    0x10(%rbx),%rax
  410771:	4c 8b 43 08          	mov    0x8(%rbx),%r8
  410775:	be d0 6a 41 00       	mov    $0x416ad0,%esi
  41077a:	4c 8b 6b 30          	mov    0x30(%rbx),%r13
  41077e:	4c 8b 63 28          	mov    0x28(%rbx),%r12
  410782:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  410786:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  41078a:	48 8b 1b             	mov    (%rbx),%rbx
  41078d:	4c 89 54 24 48       	mov    %r10,0x48(%rsp)
  410792:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
  410797:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  41079c:	4c 89 44 24 38       	mov    %r8,0x38(%rsp)
  4107a1:	31 ff                	xor    %edi,%edi
  4107a3:	e8 b8 1b ff ff       	callq  402360 <dcgettext@plt>
  4107a8:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
  4107ad:	4c 8b 54 24 48       	mov    0x48(%rsp),%r10
  4107b2:	48 89 d9             	mov    %rbx,%rcx
  4107b5:	4c 8b 44 24 38       	mov    0x38(%rsp),%r8
  4107ba:	4c 89 6c 24 18       	mov    %r13,0x18(%rsp)
  4107bf:	48 89 c2             	mov    %rax,%rdx
  4107c2:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  4107c7:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  4107cc:	48 89 ef             	mov    %rbp,%rdi
  4107cf:	4c 89 4c 24 20       	mov    %r9,0x20(%rsp)
  4107d4:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
  4107d9:	be 01 00 00 00       	mov    $0x1,%esi
  4107de:	4c 89 34 24          	mov    %r14,(%rsp)
  4107e2:	4c 89 54 24 28       	mov    %r10,0x28(%rsp)
  4107e7:	31 c0                	xor    %eax,%eax
  4107e9:	e8 22 20 ff ff       	callq  402810 <__fprintf_chk@plt>
  4107ee:	48 83 c4 58          	add    $0x58,%rsp
  4107f2:	5b                   	pop    %rbx
  4107f3:	5d                   	pop    %rbp
  4107f4:	41 5c                	pop    %r12
  4107f6:	41 5d                	pop    %r13
  4107f8:	41 5e                	pop    %r14
  4107fa:	41 5f                	pop    %r15
  4107fc:	c3                   	retq   
  4107fd:	0f 1f 00             	nopl   (%rax)
  410800:	e8 1b 1a ff ff       	callq  402220 <abort@plt>
  410805:	0f 1f 00             	nopl   (%rax)
  410808:	48 8b 1b             	mov    (%rbx),%rbx
  41080b:	ba 05 00 00 00       	mov    $0x5,%edx
  410810:	be d7 68 41 00       	mov    $0x4168d7,%esi
  410815:	31 ff                	xor    %edi,%edi
  410817:	e8 44 1b ff ff       	callq  402360 <dcgettext@plt>
  41081c:	48 83 c4 58          	add    $0x58,%rsp
  410820:	48 89 d9             	mov    %rbx,%rcx
  410823:	48 89 ef             	mov    %rbp,%rdi
  410826:	5b                   	pop    %rbx
  410827:	5d                   	pop    %rbp
  410828:	41 5c                	pop    %r12
  41082a:	41 5d                	pop    %r13
  41082c:	41 5e                	pop    %r14
  41082e:	41 5f                	pop    %r15
  410830:	48 89 c2             	mov    %rax,%rdx
  410833:	be 01 00 00 00       	mov    $0x1,%esi
  410838:	31 c0                	xor    %eax,%eax
  41083a:	e9 d1 1f ff ff       	jmpq   402810 <__fprintf_chk@plt>
  41083f:	90                   	nop
  410840:	4c 8b 63 08          	mov    0x8(%rbx),%r12
  410844:	48 8b 1b             	mov    (%rbx),%rbx
  410847:	ba 05 00 00 00       	mov    $0x5,%edx
  41084c:	be e7 68 41 00       	mov    $0x4168e7,%esi
  410851:	31 ff                	xor    %edi,%edi
  410853:	e8 08 1b ff ff       	callq  402360 <dcgettext@plt>
  410858:	48 83 c4 58          	add    $0x58,%rsp
  41085c:	48 89 d9             	mov    %rbx,%rcx
  41085f:	48 89 ef             	mov    %rbp,%rdi
  410862:	5b                   	pop    %rbx
  410863:	5d                   	pop    %rbp
  410864:	4d 89 e0             	mov    %r12,%r8
  410867:	48 89 c2             	mov    %rax,%rdx
  41086a:	be 01 00 00 00       	mov    $0x1,%esi
  41086f:	41 5c                	pop    %r12
  410871:	41 5d                	pop    %r13
  410873:	41 5e                	pop    %r14
  410875:	41 5f                	pop    %r15
  410877:	31 c0                	xor    %eax,%eax
  410879:	e9 92 1f ff ff       	jmpq   402810 <__fprintf_chk@plt>
  41087e:	66 90                	xchg   %ax,%ax
  410880:	4c 8b 6b 10          	mov    0x10(%rbx),%r13
  410884:	4c 8b 63 08          	mov    0x8(%rbx),%r12
  410888:	ba 05 00 00 00       	mov    $0x5,%edx
  41088d:	48 8b 1b             	mov    (%rbx),%rbx
  410890:	be fe 68 41 00       	mov    $0x4168fe,%esi
  410895:	31 ff                	xor    %edi,%edi
  410897:	e8 c4 1a ff ff       	callq  402360 <dcgettext@plt>
  41089c:	48 83 c4 58          	add    $0x58,%rsp
  4108a0:	48 89 ef             	mov    %rbp,%rdi
  4108a3:	4d 89 e0             	mov    %r12,%r8
  4108a6:	48 89 d9             	mov    %rbx,%rcx
  4108a9:	4d 89 e9             	mov    %r13,%r9
  4108ac:	48 89 c2             	mov    %rax,%rdx
  4108af:	5b                   	pop    %rbx
  4108b0:	5d                   	pop    %rbp
  4108b1:	41 5c                	pop    %r12
  4108b3:	41 5d                	pop    %r13
  4108b5:	41 5e                	pop    %r14
  4108b7:	41 5f                	pop    %r15
  4108b9:	be 01 00 00 00       	mov    $0x1,%esi
  4108be:	31 c0                	xor    %eax,%eax
  4108c0:	e9 4b 1f ff ff       	jmpq   402810 <__fprintf_chk@plt>
  4108c5:	0f 1f 00             	nopl   (%rax)
  4108c8:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  4108cc:	4c 8b 6b 10          	mov    0x10(%rbx),%r13
  4108d0:	31 ff                	xor    %edi,%edi
  4108d2:	4c 8b 63 08          	mov    0x8(%rbx),%r12
  4108d6:	48 8b 1b             	mov    (%rbx),%rbx
  4108d9:	ba 05 00 00 00       	mov    $0x5,%edx
  4108de:	be 00 6a 41 00       	mov    $0x416a00,%esi
  4108e3:	e8 78 1a ff ff       	callq  402360 <dcgettext@plt>
  4108e8:	4c 89 34 24          	mov    %r14,(%rsp)
  4108ec:	48 89 c2             	mov    %rax,%rdx
  4108ef:	4d 89 e9             	mov    %r13,%r9
  4108f2:	4d 89 e0             	mov    %r12,%r8
  4108f5:	48 89 d9             	mov    %rbx,%rcx
  4108f8:	be 01 00 00 00       	mov    $0x1,%esi
  4108fd:	48 89 ef             	mov    %rbp,%rdi
  410900:	31 c0                	xor    %eax,%eax
  410902:	e8 09 1f ff ff       	callq  402810 <__fprintf_chk@plt>
  410907:	e9 3f fe ff ff       	jmpq   41074b <__sprintf_chk@plt+0xdebb>
  41090c:	0f 1f 40 00          	nopl   0x0(%rax)
  410910:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  410914:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  410918:	31 ff                	xor    %edi,%edi
  41091a:	4c 8b 6b 10          	mov    0x10(%rbx),%r13
  41091e:	4c 8b 63 08          	mov    0x8(%rbx),%r12
  410922:	ba 05 00 00 00       	mov    $0x5,%edx
  410927:	48 8b 1b             	mov    (%rbx),%rbx
  41092a:	be 20 6a 41 00       	mov    $0x416a20,%esi
  41092f:	e8 2c 1a ff ff       	callq  402360 <dcgettext@plt>
  410934:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  410939:	48 89 c2             	mov    %rax,%rdx
  41093c:	4c 89 34 24          	mov    %r14,(%rsp)
  410940:	4d 89 e9             	mov    %r13,%r9
  410943:	4d 89 e0             	mov    %r12,%r8
  410946:	48 89 d9             	mov    %rbx,%rcx
  410949:	be 01 00 00 00       	mov    $0x1,%esi
  41094e:	48 89 ef             	mov    %rbp,%rdi
  410951:	31 c0                	xor    %eax,%eax
  410953:	e8 b8 1e ff ff       	callq  402810 <__fprintf_chk@plt>
  410958:	e9 ee fd ff ff       	jmpq   41074b <__sprintf_chk@plt+0xdebb>
  41095d:	0f 1f 00             	nopl   (%rax)
  410960:	4c 8b 43 08          	mov    0x8(%rbx),%r8
  410964:	4c 8b 63 28          	mov    0x28(%rbx),%r12
  410968:	31 ff                	xor    %edi,%edi
  41096a:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  41096e:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  410972:	ba 05 00 00 00       	mov    $0x5,%edx
  410977:	4c 8b 6b 10          	mov    0x10(%rbx),%r13
  41097b:	be 48 6a 41 00       	mov    $0x416a48,%esi
  410980:	48 8b 1b             	mov    (%rbx),%rbx
  410983:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
  410988:	e8 d3 19 ff ff       	callq  402360 <dcgettext@plt>
  41098d:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
  410992:	48 89 c2             	mov    %rax,%rdx
  410995:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  41099a:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  41099f:	4c 89 34 24          	mov    %r14,(%rsp)
  4109a3:	4d 89 e9             	mov    %r13,%r9
  4109a6:	48 89 d9             	mov    %rbx,%rcx
  4109a9:	be 01 00 00 00       	mov    $0x1,%esi
  4109ae:	48 89 ef             	mov    %rbp,%rdi
  4109b1:	31 c0                	xor    %eax,%eax
  4109b3:	e8 58 1e ff ff       	callq  402810 <__fprintf_chk@plt>
  4109b8:	e9 8e fd ff ff       	jmpq   41074b <__sprintf_chk@plt+0xdebb>
  4109bd:	0f 1f 00             	nopl   (%rax)
  4109c0:	4c 8b 4b 10          	mov    0x10(%rbx),%r9
  4109c4:	4c 8b 43 08          	mov    0x8(%rbx),%r8
  4109c8:	31 ff                	xor    %edi,%edi
  4109ca:	4c 8b 6b 30          	mov    0x30(%rbx),%r13
  4109ce:	4c 8b 63 28          	mov    0x28(%rbx),%r12
  4109d2:	ba 05 00 00 00       	mov    $0x5,%edx
  4109d7:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  4109db:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  4109df:	be 70 6a 41 00       	mov    $0x416a70,%esi
  4109e4:	48 8b 1b             	mov    (%rbx),%rbx
  4109e7:	4c 89 4c 24 38       	mov    %r9,0x38(%rsp)
  4109ec:	4c 89 44 24 30       	mov    %r8,0x30(%rsp)
  4109f1:	e8 6a 19 ff ff       	callq  402360 <dcgettext@plt>
  4109f6:	4c 8b 4c 24 38       	mov    0x38(%rsp),%r9
  4109fb:	4c 8b 44 24 30       	mov    0x30(%rsp),%r8
  410a00:	48 89 c2             	mov    %rax,%rdx
  410a03:	4c 89 6c 24 18       	mov    %r13,0x18(%rsp)
  410a08:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  410a0d:	48 89 d9             	mov    %rbx,%rcx
  410a10:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
  410a15:	4c 89 34 24          	mov    %r14,(%rsp)
  410a19:	be 01 00 00 00       	mov    $0x1,%esi
  410a1e:	48 89 ef             	mov    %rbp,%rdi
  410a21:	31 c0                	xor    %eax,%eax
  410a23:	e8 e8 1d ff ff       	callq  402810 <__fprintf_chk@plt>
  410a28:	e9 1e fd ff ff       	jmpq   41074b <__sprintf_chk@plt+0xdebb>
  410a2d:	0f 1f 00             	nopl   (%rax)
  410a30:	49 89 c8             	mov    %rcx,%r8
  410a33:	be 01 00 00 00       	mov    $0x1,%esi
  410a38:	48 89 d1             	mov    %rdx,%rcx
  410a3b:	31 c0                	xor    %eax,%eax
  410a3d:	ba cc 68 41 00       	mov    $0x4168cc,%edx
  410a42:	e8 c9 1d ff ff       	callq  402810 <__fprintf_chk@plt>
  410a47:	e9 1e fc ff ff       	jmpq   41066a <__sprintf_chk@plt+0xddda>
  410a4c:	0f 1f 40 00          	nopl   0x0(%rax)
  410a50:	4c 8b 53 40          	mov    0x40(%rbx),%r10
  410a54:	4c 8b 4b 38          	mov    0x38(%rbx),%r9
  410a58:	ba 05 00 00 00       	mov    $0x5,%edx
  410a5d:	48 8b 43 10          	mov    0x10(%rbx),%rax
  410a61:	4c 8b 43 08          	mov    0x8(%rbx),%r8
  410a65:	be 08 6b 41 00       	mov    $0x416b08,%esi
  410a6a:	4c 8b 6b 30          	mov    0x30(%rbx),%r13
  410a6e:	4c 8b 63 28          	mov    0x28(%rbx),%r12
  410a72:	4c 8b 7b 20          	mov    0x20(%rbx),%r15
  410a76:	4c 8b 73 18          	mov    0x18(%rbx),%r14
  410a7a:	4c 89 54 24 48       	mov    %r10,0x48(%rsp)
  410a7f:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
  410a84:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  410a89:	4c 89 44 24 38       	mov    %r8,0x38(%rsp)
  410a8e:	48 8b 1b             	mov    (%rbx),%rbx
  410a91:	e9 0b fd ff ff       	jmpq   4107a1 <__sprintf_chk@plt+0xdf11>
  410a96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  410a9d:	00 00 00 
  410aa0:	45 31 c9             	xor    %r9d,%r9d
  410aa3:	49 83 38 00          	cmpq   $0x0,(%r8)
  410aa7:	74 12                	je     410abb <__sprintf_chk@plt+0xe22b>
  410aa9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  410ab0:	49 83 c1 01          	add    $0x1,%r9
  410ab4:	4b 83 3c c8 00       	cmpq   $0x0,(%r8,%r9,8)
  410ab9:	75 f5                	jne    410ab0 <__sprintf_chk@plt+0xe220>
  410abb:	e9 70 fb ff ff       	jmpq   410630 <__sprintf_chk@plt+0xdda0>
  410ac0:	48 83 ec 58          	sub    $0x58,%rsp
  410ac4:	45 31 c9             	xor    %r9d,%r9d
  410ac7:	eb 2a                	jmp    410af3 <__sprintf_chk@plt+0xe263>
  410ac9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  410ad0:	41 89 c2             	mov    %eax,%r10d
  410ad3:	4d 03 50 10          	add    0x10(%r8),%r10
  410ad7:	83 c0 08             	add    $0x8,%eax
  410ada:	41 89 00             	mov    %eax,(%r8)
  410add:	49 8b 02             	mov    (%r10),%rax
  410ae0:	48 85 c0             	test   %rax,%rax
  410ae3:	4a 89 04 cc          	mov    %rax,(%rsp,%r9,8)
  410ae7:	74 2f                	je     410b18 <__sprintf_chk@plt+0xe288>
  410ae9:	49 83 c1 01          	add    $0x1,%r9
  410aed:	49 83 f9 0a          	cmp    $0xa,%r9
  410af1:	74 25                	je     410b18 <__sprintf_chk@plt+0xe288>
  410af3:	41 8b 00             	mov    (%r8),%eax
  410af6:	83 f8 30             	cmp    $0x30,%eax
  410af9:	72 d5                	jb     410ad0 <__sprintf_chk@plt+0xe240>
  410afb:	4d 8b 50 08          	mov    0x8(%r8),%r10
  410aff:	49 8d 42 08          	lea    0x8(%r10),%rax
  410b03:	49 89 40 08          	mov    %rax,0x8(%r8)
  410b07:	49 8b 02             	mov    (%r10),%rax
  410b0a:	48 85 c0             	test   %rax,%rax
  410b0d:	4a 89 04 cc          	mov    %rax,(%rsp,%r9,8)
  410b11:	75 d6                	jne    410ae9 <__sprintf_chk@plt+0xe259>
  410b13:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  410b18:	49 89 e0             	mov    %rsp,%r8
  410b1b:	e8 10 fb ff ff       	callq  410630 <__sprintf_chk@plt+0xdda0>
  410b20:	48 83 c4 58          	add    $0x58,%rsp
  410b24:	c3                   	retq   
  410b25:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  410b2c:	00 00 00 00 
  410b30:	48 81 ec d8 00 00 00 	sub    $0xd8,%rsp
  410b37:	84 c0                	test   %al,%al
  410b39:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
  410b3e:	4c 89 4c 24 48       	mov    %r9,0x48(%rsp)
  410b43:	74 37                	je     410b7c <__sprintf_chk@plt+0xe2ec>
  410b45:	0f 29 44 24 50       	movaps %xmm0,0x50(%rsp)
  410b4a:	0f 29 4c 24 60       	movaps %xmm1,0x60(%rsp)
  410b4f:	0f 29 54 24 70       	movaps %xmm2,0x70(%rsp)
  410b54:	0f 29 9c 24 80 00 00 	movaps %xmm3,0x80(%rsp)
  410b5b:	00 
  410b5c:	0f 29 a4 24 90 00 00 	movaps %xmm4,0x90(%rsp)
  410b63:	00 
  410b64:	0f 29 ac 24 a0 00 00 	movaps %xmm5,0xa0(%rsp)
  410b6b:	00 
  410b6c:	0f 29 b4 24 b0 00 00 	movaps %xmm6,0xb0(%rsp)
  410b73:	00 
  410b74:	0f 29 bc 24 c0 00 00 	movaps %xmm7,0xc0(%rsp)
  410b7b:	00 
  410b7c:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
  410b83:	00 
  410b84:	4c 8d 44 24 08       	lea    0x8(%rsp),%r8
  410b89:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
  410b8e:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
  410b93:	c7 44 24 08 20 00 00 	movl   $0x20,0x8(%rsp)
  410b9a:	00 
  410b9b:	c7 44 24 0c 30 00 00 	movl   $0x30,0xc(%rsp)
  410ba2:	00 
  410ba3:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  410ba8:	e8 13 ff ff ff       	callq  410ac0 <__sprintf_chk@plt+0xe230>
  410bad:	48 81 c4 d8 00 00 00 	add    $0xd8,%rsp
  410bb4:	c3                   	retq   
  410bb5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  410bbc:	00 00 00 00 
  410bc0:	53                   	push   %rbx
  410bc1:	ba 05 00 00 00       	mov    $0x5,%edx
  410bc6:	be 1a 69 41 00       	mov    $0x41691a,%esi
  410bcb:	31 ff                	xor    %edi,%edi
  410bcd:	e8 8e 17 ff ff       	callq  402360 <dcgettext@plt>
  410bd2:	ba d2 37 41 00       	mov    $0x4137d2,%edx
  410bd7:	48 89 c6             	mov    %rax,%rsi
  410bda:	bf 01 00 00 00       	mov    $0x1,%edi
  410bdf:	31 c0                	xor    %eax,%eax
  410be1:	e8 4a 1b ff ff       	callq  402730 <__printf_chk@plt>
  410be6:	ba 05 00 00 00       	mov    $0x5,%edx
  410beb:	be e8 37 41 00       	mov    $0x4137e8,%esi
  410bf0:	31 ff                	xor    %edi,%edi
  410bf2:	e8 69 17 ff ff       	callq  402360 <dcgettext@plt>
  410bf7:	b9 08 5a 41 00       	mov    $0x415a08,%ecx
  410bfc:	48 89 c6             	mov    %rax,%rsi
  410bff:	ba fc 37 41 00       	mov    $0x4137fc,%edx
  410c04:	bf 01 00 00 00       	mov    $0x1,%edi
  410c09:	31 c0                	xor    %eax,%eax
  410c0b:	e8 20 1b ff ff       	callq  402730 <__printf_chk@plt>
  410c10:	48 8b 1d f9 99 20 00 	mov    0x2099f9(%rip),%rbx        # 61a610 <stdout@@GLIBC_2.2.5>
  410c17:	be 30 5a 41 00       	mov    $0x415a30,%esi
  410c1c:	31 ff                	xor    %edi,%edi
  410c1e:	ba 05 00 00 00       	mov    $0x5,%edx
  410c23:	e8 38 17 ff ff       	callq  402360 <dcgettext@plt>
  410c28:	48 89 de             	mov    %rbx,%rsi
  410c2b:	48 89 c7             	mov    %rax,%rdi
  410c2e:	5b                   	pop    %rbx
  410c2f:	e9 ec 18 ff ff       	jmpq   402520 <fputs_unlocked@plt>
  410c34:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  410c3b:	00 00 00 
  410c3e:	66 90                	xchg   %ax,%ax
  410c40:	53                   	push   %rbx
  410c41:	48 89 fb             	mov    %rdi,%rbx
  410c44:	e8 f7 19 ff ff       	callq  402640 <malloc@plt>
  410c49:	48 85 c0             	test   %rax,%rax
  410c4c:	74 02                	je     410c50 <__sprintf_chk@plt+0xe3c0>
  410c4e:	5b                   	pop    %rbx
  410c4f:	c3                   	retq   
  410c50:	48 85 db             	test   %rbx,%rbx
  410c53:	74 f9                	je     410c4e <__sprintf_chk@plt+0xe3be>
  410c55:	e8 f6 01 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410c5a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  410c60:	31 d2                	xor    %edx,%edx
  410c62:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  410c69:	48 f7 f6             	div    %rsi
  410c6c:	48 39 f8             	cmp    %rdi,%rax
  410c6f:	72 09                	jb     410c7a <__sprintf_chk@plt+0xe3ea>
  410c71:	48 0f af fe          	imul   %rsi,%rdi
  410c75:	e9 c6 ff ff ff       	jmpq   410c40 <__sprintf_chk@plt+0xe3b0>
  410c7a:	50                   	push   %rax
  410c7b:	e8 d0 01 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410c80:	e9 bb ff ff ff       	jmpq   410c40 <__sprintf_chk@plt+0xe3b0>
  410c85:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  410c8c:	00 00 00 00 
  410c90:	48 85 f6             	test   %rsi,%rsi
  410c93:	53                   	push   %rbx
  410c94:	48 89 f3             	mov    %rsi,%rbx
  410c97:	74 17                	je     410cb0 <__sprintf_chk@plt+0xe420>
  410c99:	48 89 de             	mov    %rbx,%rsi
  410c9c:	e8 3f 1a ff ff       	callq  4026e0 <realloc@plt>
  410ca1:	48 85 c0             	test   %rax,%rax
  410ca4:	74 18                	je     410cbe <__sprintf_chk@plt+0xe42e>
  410ca6:	5b                   	pop    %rbx
  410ca7:	c3                   	retq   
  410ca8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  410caf:	00 
  410cb0:	48 85 ff             	test   %rdi,%rdi
  410cb3:	74 e4                	je     410c99 <__sprintf_chk@plt+0xe409>
  410cb5:	e8 36 15 ff ff       	callq  4021f0 <free@plt>
  410cba:	31 c0                	xor    %eax,%eax
  410cbc:	5b                   	pop    %rbx
  410cbd:	c3                   	retq   
  410cbe:	48 85 db             	test   %rbx,%rbx
  410cc1:	74 e3                	je     410ca6 <__sprintf_chk@plt+0xe416>
  410cc3:	e8 88 01 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410cc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  410ccf:	00 
  410cd0:	48 89 d1             	mov    %rdx,%rcx
  410cd3:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  410cda:	31 d2                	xor    %edx,%edx
  410cdc:	48 f7 f1             	div    %rcx
  410cdf:	48 39 f0             	cmp    %rsi,%rax
  410ce2:	72 09                	jb     410ced <__sprintf_chk@plt+0xe45d>
  410ce4:	48 0f af f1          	imul   %rcx,%rsi
  410ce8:	e9 a3 ff ff ff       	jmpq   410c90 <__sprintf_chk@plt+0xe400>
  410ced:	50                   	push   %rax
  410cee:	e8 5d 01 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410cf3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  410cfa:	84 00 00 00 00 00 
  410d00:	48 85 ff             	test   %rdi,%rdi
  410d03:	49 89 d0             	mov    %rdx,%r8
  410d06:	48 8b 0e             	mov    (%rsi),%rcx
  410d09:	74 35                	je     410d40 <__sprintf_chk@plt+0xe4b0>
  410d0b:	31 d2                	xor    %edx,%edx
  410d0d:	48 b8 aa aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaaa,%rax
  410d14:	aa aa aa 
  410d17:	49 f7 f0             	div    %r8
  410d1a:	48 39 c1             	cmp    %rax,%rcx
  410d1d:	73 3d                	jae    410d5c <__sprintf_chk@plt+0xe4cc>
  410d1f:	48 8d 41 01          	lea    0x1(%rcx),%rax
  410d23:	48 d1 e8             	shr    %rax
  410d26:	48 01 c1             	add    %rax,%rcx
  410d29:	48 89 0e             	mov    %rcx,(%rsi)
  410d2c:	49 0f af c8          	imul   %r8,%rcx
  410d30:	48 89 ce             	mov    %rcx,%rsi
  410d33:	e9 58 ff ff ff       	jmpq   410c90 <__sprintf_chk@plt+0xe400>
  410d38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  410d3f:	00 
  410d40:	48 85 c9             	test   %rcx,%rcx
  410d43:	75 e4                	jne    410d29 <__sprintf_chk@plt+0xe499>
  410d45:	31 d2                	xor    %edx,%edx
  410d47:	b8 80 00 00 00       	mov    $0x80,%eax
  410d4c:	31 c9                	xor    %ecx,%ecx
  410d4e:	49 f7 f0             	div    %r8
  410d51:	48 85 c0             	test   %rax,%rax
  410d54:	0f 94 c1             	sete   %cl
  410d57:	48 01 c1             	add    %rax,%rcx
  410d5a:	eb cd                	jmp    410d29 <__sprintf_chk@plt+0xe499>
  410d5c:	50                   	push   %rax
  410d5d:	e8 ee 00 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410d62:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  410d69:	1f 84 00 00 00 00 00 
  410d70:	48 85 ff             	test   %rdi,%rdi
  410d73:	48 8b 06             	mov    (%rsi),%rax
  410d76:	74 28                	je     410da0 <__sprintf_chk@plt+0xe510>
  410d78:	48 ba a9 aa aa aa aa 	movabs $0xaaaaaaaaaaaaaaa9,%rdx
  410d7f:	aa aa aa 
  410d82:	48 39 d0             	cmp    %rdx,%rax
  410d85:	77 30                	ja     410db7 <__sprintf_chk@plt+0xe527>
  410d87:	48 8d 50 01          	lea    0x1(%rax),%rdx
  410d8b:	48 d1 ea             	shr    %rdx
  410d8e:	48 01 d0             	add    %rdx,%rax
  410d91:	48 89 06             	mov    %rax,(%rsi)
  410d94:	48 89 c6             	mov    %rax,%rsi
  410d97:	e9 f4 fe ff ff       	jmpq   410c90 <__sprintf_chk@plt+0xe400>
  410d9c:	0f 1f 40 00          	nopl   0x0(%rax)
  410da0:	48 85 c0             	test   %rax,%rax
  410da3:	ba 80 00 00 00       	mov    $0x80,%edx
  410da8:	48 0f 44 c2          	cmove  %rdx,%rax
  410dac:	48 89 06             	mov    %rax,(%rsi)
  410daf:	48 89 c6             	mov    %rax,%rsi
  410db2:	e9 d9 fe ff ff       	jmpq   410c90 <__sprintf_chk@plt+0xe400>
  410db7:	50                   	push   %rax
  410db8:	e8 93 00 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410dbd:	0f 1f 00             	nopl   (%rax)
  410dc0:	53                   	push   %rbx
  410dc1:	48 89 fb             	mov    %rdi,%rbx
  410dc4:	e8 77 fe ff ff       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  410dc9:	48 89 da             	mov    %rbx,%rdx
  410dcc:	31 f6                	xor    %esi,%esi
  410dce:	48 89 c7             	mov    %rax,%rdi
  410dd1:	5b                   	pop    %rbx
  410dd2:	e9 a9 16 ff ff       	jmpq   402480 <memset@plt>
  410dd7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  410dde:	00 00 
  410de0:	48 83 ec 08          	sub    $0x8,%rsp
  410de4:	e8 47 17 ff ff       	callq  402530 <calloc@plt>
  410de9:	48 85 c0             	test   %rax,%rax
  410dec:	74 05                	je     410df3 <__sprintf_chk@plt+0xe563>
  410dee:	48 83 c4 08          	add    $0x8,%rsp
  410df2:	c3                   	retq   
  410df3:	e8 58 00 00 00       	callq  410e50 <__sprintf_chk@plt+0xe5c0>
  410df8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  410dff:	00 
  410e00:	55                   	push   %rbp
  410e01:	48 89 fd             	mov    %rdi,%rbp
  410e04:	48 89 f7             	mov    %rsi,%rdi
  410e07:	53                   	push   %rbx
  410e08:	48 89 f3             	mov    %rsi,%rbx
  410e0b:	48 83 ec 08          	sub    $0x8,%rsp
  410e0f:	e8 2c fe ff ff       	callq  410c40 <__sprintf_chk@plt+0xe3b0>
  410e14:	48 83 c4 08          	add    $0x8,%rsp
  410e18:	48 89 da             	mov    %rbx,%rdx
  410e1b:	48 89 ee             	mov    %rbp,%rsi
  410e1e:	5b                   	pop    %rbx
  410e1f:	5d                   	pop    %rbp
  410e20:	48 89 c7             	mov    %rax,%rdi
  410e23:	e9 98 17 ff ff       	jmpq   4025c0 <memcpy@plt>
  410e28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  410e2f:	00 
  410e30:	53                   	push   %rbx
  410e31:	48 89 fb             	mov    %rdi,%rbx
  410e34:	e8 47 15 ff ff       	callq  402380 <strlen@plt>
  410e39:	48 89 df             	mov    %rbx,%rdi
  410e3c:	48 8d 70 01          	lea    0x1(%rax),%rsi
  410e40:	5b                   	pop    %rbx
  410e41:	e9 ba ff ff ff       	jmpq   410e00 <__sprintf_chk@plt+0xe570>
  410e46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  410e4d:	00 00 00 
  410e50:	48 83 ec 08          	sub    $0x8,%rsp
  410e54:	ba 05 00 00 00       	mov    $0x5,%edx
  410e59:	be cf 6b 41 00       	mov    $0x416bcf,%esi
  410e5e:	31 ff                	xor    %edi,%edi
  410e60:	e8 fb 14 ff ff       	callq  402360 <dcgettext@plt>
  410e65:	8b 3d 15 97 20 00    	mov    0x209715(%rip),%edi        # 61a580 <_fini@@Base+0x208684>
  410e6b:	48 89 c1             	mov    %rax,%rcx
  410e6e:	ba 54 5e 41 00       	mov    $0x415e54,%edx
  410e73:	31 f6                	xor    %esi,%esi
  410e75:	31 c0                	xor    %eax,%eax
  410e77:	e8 f4 18 ff ff       	callq  402770 <error@plt>
  410e7c:	e8 9f 13 ff ff       	callq  402220 <abort@plt>
  410e81:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  410e88:	00 00 00 
  410e8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  410e90:	41 57                	push   %r15
  410e92:	41 56                	push   %r14
  410e94:	41 89 d6             	mov    %edx,%r14d
  410e97:	41 55                	push   %r13
  410e99:	41 54                	push   %r12
  410e9b:	55                   	push   %rbp
  410e9c:	53                   	push   %rbx
  410e9d:	48 83 ec 28          	sub    $0x28,%rsp
  410ea1:	83 fa 24             	cmp    $0x24,%edx
  410ea4:	0f 87 06 04 00 00    	ja     4112b0 <__sprintf_chk@plt+0xea20>
  410eaa:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
  410eaf:	48 89 fd             	mov    %rdi,%rbp
  410eb2:	49 89 f7             	mov    %rsi,%r15
  410eb5:	48 85 f6             	test   %rsi,%rsi
  410eb8:	49 89 cd             	mov    %rcx,%r13
  410ebb:	4d 89 c4             	mov    %r8,%r12
  410ebe:	4c 0f 44 f8          	cmove  %rax,%r15
  410ec2:	0f b6 1f             	movzbl (%rdi),%ebx
  410ec5:	e8 b6 19 ff ff       	callq  402880 <__ctype_b_loc@plt>
  410eca:	48 8b 10             	mov    (%rax),%rdx
  410ecd:	48 89 e8             	mov    %rbp,%rax
  410ed0:	eb 0d                	jmp    410edf <__sprintf_chk@plt+0xe64f>
  410ed2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  410ed8:	48 83 c0 01          	add    $0x1,%rax
  410edc:	0f b6 18             	movzbl (%rax),%ebx
  410edf:	44 0f b6 cb          	movzbl %bl,%r9d
  410ee3:	42 f6 44 4a 01 20    	testb  $0x20,0x1(%rdx,%r9,2)
  410ee9:	75 ed                	jne    410ed8 <__sprintf_chk@plt+0xe648>
  410eeb:	80 fb 2d             	cmp    $0x2d,%bl
  410eee:	75 18                	jne    410f08 <__sprintf_chk@plt+0xe678>
  410ef0:	b8 04 00 00 00       	mov    $0x4,%eax
  410ef5:	48 83 c4 28          	add    $0x28,%rsp
  410ef9:	5b                   	pop    %rbx
  410efa:	5d                   	pop    %rbp
  410efb:	41 5c                	pop    %r12
  410efd:	41 5d                	pop    %r13
  410eff:	41 5e                	pop    %r14
  410f01:	41 5f                	pop    %r15
  410f03:	c3                   	retq   
  410f04:	0f 1f 40 00          	nopl   0x0(%rax)
  410f08:	e8 23 13 ff ff       	callq  402230 <__errno_location@plt>
  410f0d:	44 89 f2             	mov    %r14d,%edx
  410f10:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  410f16:	4c 89 fe             	mov    %r15,%rsi
  410f19:	48 89 ef             	mov    %rbp,%rdi
  410f1c:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  410f21:	e8 7a 18 ff ff       	callq  4027a0 <strtoul@plt>
  410f26:	4d 8b 37             	mov    (%r15),%r14
  410f29:	48 89 c3             	mov    %rax,%rbx
  410f2c:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
  410f31:	49 39 ee             	cmp    %rbp,%r14
  410f34:	0f 84 ad 00 00 00    	je     410fe7 <__sprintf_chk@plt+0xe757>
  410f3a:	8b 01                	mov    (%rcx),%eax
  410f3c:	85 c0                	test   %eax,%eax
  410f3e:	75 28                	jne    410f68 <__sprintf_chk@plt+0xe6d8>
  410f40:	31 ed                	xor    %ebp,%ebp
  410f42:	4d 85 e4             	test   %r12,%r12
  410f45:	74 08                	je     410f4f <__sprintf_chk@plt+0xe6bf>
  410f47:	41 0f b6 16          	movzbl (%r14),%edx
  410f4b:	84 d2                	test   %dl,%dl
  410f4d:	75 29                	jne    410f78 <__sprintf_chk@plt+0xe6e8>
  410f4f:	49 89 5d 00          	mov    %rbx,0x0(%r13)
  410f53:	48 83 c4 28          	add    $0x28,%rsp
  410f57:	89 e8                	mov    %ebp,%eax
  410f59:	5b                   	pop    %rbx
  410f5a:	5d                   	pop    %rbp
  410f5b:	41 5c                	pop    %r12
  410f5d:	41 5d                	pop    %r13
  410f5f:	41 5e                	pop    %r14
  410f61:	41 5f                	pop    %r15
  410f63:	c3                   	retq   
  410f64:	0f 1f 40 00          	nopl   0x0(%rax)
  410f68:	83 f8 22             	cmp    $0x22,%eax
  410f6b:	bd 01 00 00 00       	mov    $0x1,%ebp
  410f70:	0f 85 7a ff ff ff    	jne    410ef0 <__sprintf_chk@plt+0xe660>
  410f76:	eb ca                	jmp    410f42 <__sprintf_chk@plt+0xe6b2>
  410f78:	0f be f2             	movsbl %dl,%esi
  410f7b:	4c 89 e7             	mov    %r12,%rdi
  410f7e:	89 54 24 08          	mov    %edx,0x8(%rsp)
  410f82:	e8 49 14 ff ff       	callq  4023d0 <strchr@plt>
  410f87:	48 85 c0             	test   %rax,%rax
  410f8a:	8b 54 24 08          	mov    0x8(%rsp),%edx
  410f8e:	0f 84 94 00 00 00    	je     411028 <__sprintf_chk@plt+0xe798>
  410f94:	be 30 00 00 00       	mov    $0x30,%esi
  410f99:	4c 89 e7             	mov    %r12,%rdi
  410f9c:	89 54 24 08          	mov    %edx,0x8(%rsp)
  410fa0:	e8 2b 14 ff ff       	callq  4023d0 <strchr@plt>
  410fa5:	48 85 c0             	test   %rax,%rax
  410fa8:	8b 54 24 08          	mov    0x8(%rsp),%edx
  410fac:	74 1d                	je     410fcb <__sprintf_chk@plt+0xe73b>
  410fae:	41 0f b6 46 01       	movzbl 0x1(%r14),%eax
  410fb3:	3c 44                	cmp    $0x44,%al
  410fb5:	0f 84 a7 02 00 00    	je     411262 <__sprintf_chk@plt+0xe9d2>
  410fbb:	3c 69                	cmp    $0x69,%al
  410fbd:	0f 84 87 02 00 00    	je     41124a <__sprintf_chk@plt+0xe9ba>
  410fc3:	3c 42                	cmp    $0x42,%al
  410fc5:	0f 84 97 02 00 00    	je     411262 <__sprintf_chk@plt+0xe9d2>
  410fcb:	b9 01 00 00 00       	mov    $0x1,%ecx
  410fd0:	b8 00 04 00 00       	mov    $0x400,%eax
  410fd5:	83 ea 42             	sub    $0x42,%edx
  410fd8:	80 fa 35             	cmp    $0x35,%dl
  410fdb:	77 4b                	ja     411028 <__sprintf_chk@plt+0xe798>
  410fdd:	0f b6 d2             	movzbl %dl,%edx
  410fe0:	ff 24 d5 18 6c 41 00 	jmpq   *0x416c18(,%rdx,8)
  410fe7:	4d 85 e4             	test   %r12,%r12
  410fea:	0f 84 00 ff ff ff    	je     410ef0 <__sprintf_chk@plt+0xe660>
  410ff0:	0f b6 55 00          	movzbl 0x0(%rbp),%edx
  410ff4:	84 d2                	test   %dl,%dl
  410ff6:	0f 84 f4 fe ff ff    	je     410ef0 <__sprintf_chk@plt+0xe660>
  410ffc:	0f be f2             	movsbl %dl,%esi
  410fff:	4c 89 e7             	mov    %r12,%rdi
  411002:	89 54 24 08          	mov    %edx,0x8(%rsp)
  411006:	31 ed                	xor    %ebp,%ebp
  411008:	bb 01 00 00 00       	mov    $0x1,%ebx
  41100d:	e8 be 13 ff ff       	callq  4023d0 <strchr@plt>
  411012:	48 85 c0             	test   %rax,%rax
  411015:	8b 54 24 08          	mov    0x8(%rsp),%edx
  411019:	0f 85 75 ff ff ff    	jne    410f94 <__sprintf_chk@plt+0xe704>
  41101f:	e9 cc fe ff ff       	jmpq   410ef0 <__sprintf_chk@plt+0xe660>
  411024:	0f 1f 40 00          	nopl   0x0(%rax)
  411028:	89 e8                	mov    %ebp,%eax
  41102a:	49 89 5d 00          	mov    %rbx,0x0(%r13)
  41102e:	83 c8 02             	or     $0x2,%eax
  411031:	e9 bf fe ff ff       	jmpq   410ef5 <__sprintf_chk@plt+0xe665>
  411036:	31 d2                	xor    %edx,%edx
  411038:	09 d5                	or     %edx,%ebp
  41103a:	48 63 c9             	movslq %ecx,%rcx
  41103d:	49 01 ce             	add    %rcx,%r14
  411040:	89 e8                	mov    %ebp,%eax
  411042:	83 c8 02             	or     $0x2,%eax
  411045:	4d 89 37             	mov    %r14,(%r15)
  411048:	41 80 3e 00          	cmpb   $0x0,(%r14)
  41104c:	0f 45 e8             	cmovne %eax,%ebp
  41104f:	e9 fb fe ff ff       	jmpq   410f4f <__sprintf_chk@plt+0xe6bf>
  411054:	48 85 db             	test   %rbx,%rbx
  411057:	0f 88 26 02 00 00    	js     411283 <__sprintf_chk@plt+0xe9f3>
  41105d:	48 01 db             	add    %rbx,%rbx
  411060:	31 d2                	xor    %edx,%edx
  411062:	eb d4                	jmp    411038 <__sprintf_chk@plt+0xe7a8>
  411064:	48 b8 ff ff ff ff ff 	movabs $0x7fffffffffffff,%rax
  41106b:	ff 7f 00 
  41106e:	48 39 c3             	cmp    %rax,%rbx
  411071:	0f 87 0c 02 00 00    	ja     411283 <__sprintf_chk@plt+0xe9f3>
  411077:	48 c1 e3 09          	shl    $0x9,%rbx
  41107b:	31 d2                	xor    %edx,%edx
  41107d:	eb b9                	jmp    411038 <__sprintf_chk@plt+0xe7a8>
  41107f:	48 63 f8             	movslq %eax,%rdi
  411082:	31 d2                	xor    %edx,%edx
  411084:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  41108b:	48 f7 f7             	div    %rdi
  41108e:	be 07 00 00 00       	mov    $0x7,%esi
  411093:	31 d2                	xor    %edx,%edx
  411095:	eb 0f                	jmp    4110a6 <__sprintf_chk@plt+0xe816>
  411097:	48 0f af df          	imul   %rdi,%rbx
  41109b:	45 31 c0             	xor    %r8d,%r8d
  41109e:	44 09 c2             	or     %r8d,%edx
  4110a1:	83 ee 01             	sub    $0x1,%esi
  4110a4:	74 92                	je     411038 <__sprintf_chk@plt+0xe7a8>
  4110a6:	48 39 d8             	cmp    %rbx,%rax
  4110a9:	73 ec                	jae    411097 <__sprintf_chk@plt+0xe807>
  4110ab:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  4110b2:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4110b8:	eb e4                	jmp    41109e <__sprintf_chk@plt+0xe80e>
  4110ba:	48 63 f8             	movslq %eax,%rdi
  4110bd:	31 d2                	xor    %edx,%edx
  4110bf:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4110c6:	48 f7 f7             	div    %rdi
  4110c9:	be 08 00 00 00       	mov    $0x8,%esi
  4110ce:	31 d2                	xor    %edx,%edx
  4110d0:	eb 13                	jmp    4110e5 <__sprintf_chk@plt+0xe855>
  4110d2:	48 0f af df          	imul   %rdi,%rbx
  4110d6:	45 31 c0             	xor    %r8d,%r8d
  4110d9:	44 09 c2             	or     %r8d,%edx
  4110dc:	83 ee 01             	sub    $0x1,%esi
  4110df:	0f 84 53 ff ff ff    	je     411038 <__sprintf_chk@plt+0xe7a8>
  4110e5:	48 39 d8             	cmp    %rbx,%rax
  4110e8:	73 e8                	jae    4110d2 <__sprintf_chk@plt+0xe842>
  4110ea:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  4110f1:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4110f7:	eb e0                	jmp    4110d9 <__sprintf_chk@plt+0xe849>
  4110f9:	48 63 f0             	movslq %eax,%rsi
  4110fc:	31 d2                	xor    %edx,%edx
  4110fe:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  411105:	48 f7 f6             	div    %rsi
  411108:	bf 04 00 00 00       	mov    $0x4,%edi
  41110d:	31 d2                	xor    %edx,%edx
  41110f:	48 39 d8             	cmp    %rbx,%rax
  411112:	0f 82 59 01 00 00    	jb     411271 <__sprintf_chk@plt+0xe9e1>
  411118:	45 31 c0             	xor    %r8d,%r8d
  41111b:	48 0f af de          	imul   %rsi,%rbx
  41111f:	44 09 c2             	or     %r8d,%edx
  411122:	83 ef 01             	sub    $0x1,%edi
  411125:	75 e8                	jne    41110f <__sprintf_chk@plt+0xe87f>
  411127:	e9 0c ff ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  41112c:	48 63 f0             	movslq %eax,%rsi
  41112f:	31 d2                	xor    %edx,%edx
  411131:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  411138:	48 f7 f6             	div    %rsi
  41113b:	bf 05 00 00 00       	mov    $0x5,%edi
  411140:	31 d2                	xor    %edx,%edx
  411142:	eb 13                	jmp    411157 <__sprintf_chk@plt+0xe8c7>
  411144:	48 0f af de          	imul   %rsi,%rbx
  411148:	45 31 c0             	xor    %r8d,%r8d
  41114b:	44 09 c2             	or     %r8d,%edx
  41114e:	83 ef 01             	sub    $0x1,%edi
  411151:	0f 84 e1 fe ff ff    	je     411038 <__sprintf_chk@plt+0xe7a8>
  411157:	48 39 d8             	cmp    %rbx,%rax
  41115a:	73 e8                	jae    411144 <__sprintf_chk@plt+0xe8b4>
  41115c:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  411163:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  411169:	eb e0                	jmp    41114b <__sprintf_chk@plt+0xe8bb>
  41116b:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  411172:	48 63 f8             	movslq %eax,%rdi
  411175:	31 d2                	xor    %edx,%edx
  411177:	48 89 f0             	mov    %rsi,%rax
  41117a:	48 f7 f7             	div    %rdi
  41117d:	48 39 c3             	cmp    %rax,%rbx
  411180:	0f 87 1d 01 00 00    	ja     4112a3 <__sprintf_chk@plt+0xea13>
  411186:	48 0f af df          	imul   %rdi,%rbx
  41118a:	48 39 d8             	cmp    %rbx,%rax
  41118d:	0f 82 10 01 00 00    	jb     4112a3 <__sprintf_chk@plt+0xea13>
  411193:	48 0f af df          	imul   %rdi,%rbx
  411197:	31 d2                	xor    %edx,%edx
  411199:	e9 9a fe ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  41119e:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  4111a5:	48 63 f8             	movslq %eax,%rdi
  4111a8:	31 d2                	xor    %edx,%edx
  4111aa:	48 89 f0             	mov    %rsi,%rax
  4111ad:	48 f7 f7             	div    %rdi
  4111b0:	48 39 c3             	cmp    %rax,%rbx
  4111b3:	76 de                	jbe    411193 <__sprintf_chk@plt+0xe903>
  4111b5:	48 89 f3             	mov    %rsi,%rbx
  4111b8:	ba 01 00 00 00       	mov    $0x1,%edx
  4111bd:	e9 76 fe ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  4111c2:	48 63 f8             	movslq %eax,%rdi
  4111c5:	31 d2                	xor    %edx,%edx
  4111c7:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4111ce:	48 f7 f7             	div    %rdi
  4111d1:	be 06 00 00 00       	mov    $0x6,%esi
  4111d6:	31 d2                	xor    %edx,%edx
  4111d8:	eb 13                	jmp    4111ed <__sprintf_chk@plt+0xe95d>
  4111da:	48 0f af df          	imul   %rdi,%rbx
  4111de:	45 31 c0             	xor    %r8d,%r8d
  4111e1:	44 09 c2             	or     %r8d,%edx
  4111e4:	83 ee 01             	sub    $0x1,%esi
  4111e7:	0f 84 4b fe ff ff    	je     411038 <__sprintf_chk@plt+0xe7a8>
  4111ed:	48 39 d8             	cmp    %rbx,%rax
  4111f0:	73 e8                	jae    4111da <__sprintf_chk@plt+0xe94a>
  4111f2:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  4111f9:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4111ff:	eb e0                	jmp    4111e1 <__sprintf_chk@plt+0xe951>
  411201:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffff,%rax
  411208:	ff 3f 00 
  41120b:	48 39 c3             	cmp    %rax,%rbx
  41120e:	77 73                	ja     411283 <__sprintf_chk@plt+0xe9f3>
  411210:	48 c1 e3 0a          	shl    $0xa,%rbx
  411214:	31 d2                	xor    %edx,%edx
  411216:	e9 1d fe ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  41121b:	48 63 f0             	movslq %eax,%rsi
  41121e:	31 d2                	xor    %edx,%edx
  411220:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  411227:	48 f7 f6             	div    %rsi
  41122a:	bf 03 00 00 00       	mov    $0x3,%edi
  41122f:	31 d2                	xor    %edx,%edx
  411231:	48 39 d8             	cmp    %rbx,%rax
  411234:	72 5e                	jb     411294 <__sprintf_chk@plt+0xea04>
  411236:	45 31 c0             	xor    %r8d,%r8d
  411239:	48 0f af de          	imul   %rsi,%rbx
  41123d:	44 09 c2             	or     %r8d,%edx
  411240:	83 ef 01             	sub    $0x1,%edi
  411243:	75 ec                	jne    411231 <__sprintf_chk@plt+0xe9a1>
  411245:	e9 ee fd ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  41124a:	31 c9                	xor    %ecx,%ecx
  41124c:	41 80 7e 02 42       	cmpb   $0x42,0x2(%r14)
  411251:	b8 00 04 00 00       	mov    $0x400,%eax
  411256:	0f 94 c1             	sete   %cl
  411259:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
  41125d:	e9 73 fd ff ff       	jmpq   410fd5 <__sprintf_chk@plt+0xe745>
  411262:	b9 02 00 00 00       	mov    $0x2,%ecx
  411267:	b8 e8 03 00 00       	mov    $0x3e8,%eax
  41126c:	e9 64 fd ff ff       	jmpq   410fd5 <__sprintf_chk@plt+0xe745>
  411271:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  411278:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  41127e:	e9 9c fe ff ff       	jmpq   41111f <__sprintf_chk@plt+0xe88f>
  411283:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  41128a:	ba 01 00 00 00       	mov    $0x1,%edx
  41128f:	e9 a4 fd ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  411294:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  41129b:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4112a1:	eb 9a                	jmp    41123d <__sprintf_chk@plt+0xe9ad>
  4112a3:	ba 01 00 00 00       	mov    $0x1,%edx
  4112a8:	48 89 f3             	mov    %rsi,%rbx
  4112ab:	e9 88 fd ff ff       	jmpq   411038 <__sprintf_chk@plt+0xe7a8>
  4112b0:	b9 c8 6d 41 00       	mov    $0x416dc8,%ecx
  4112b5:	ba 60 00 00 00       	mov    $0x60,%edx
  4112ba:	be e0 6b 41 00       	mov    $0x416be0,%esi
  4112bf:	bf f0 6b 41 00       	mov    $0x416bf0,%edi
  4112c4:	e8 87 11 ff ff       	callq  402450 <__assert_fail@plt>
  4112c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4112d0:	41 55                	push   %r13
  4112d2:	4c 63 d6             	movslq %esi,%r10
  4112d5:	41 54                	push   %r12
  4112d7:	4d 89 c4             	mov    %r8,%r12
  4112da:	55                   	push   %rbp
  4112db:	53                   	push   %rbx
  4112dc:	48 83 ec 18          	sub    $0x18,%rsp
  4112e0:	83 ff 03             	cmp    $0x3,%edi
  4112e3:	8b 2d 97 92 20 00    	mov    0x209297(%rip),%ebp        # 61a580 <_fini@@Base+0x208684>
  4112e9:	77 50                	ja     41133b <__sprintf_chk@plt+0xeaab>
  4112eb:	83 ff 02             	cmp    $0x2,%edi
  4112ee:	73 44                	jae    411334 <__sprintf_chk@plt+0xeaa4>
  4112f0:	83 ef 01             	sub    $0x1,%edi
  4112f3:	be ec 6d 41 00       	mov    $0x416dec,%esi
  4112f8:	75 35                	jne    41132f <__sprintf_chk@plt+0xea9f>
  4112fa:	45 85 d2             	test   %r10d,%r10d
  4112fd:	78 48                	js     411347 <__sprintf_chk@plt+0xeab7>
  4112ff:	49 c1 e2 05          	shl    $0x5,%r10
  411303:	bb 09 6e 41 00       	mov    $0x416e09,%ebx
  411308:	4e 8b 2c 11          	mov    (%rcx,%r10,1),%r13
  41130c:	ba 05 00 00 00       	mov    $0x5,%edx
  411311:	31 ff                	xor    %edi,%edi
  411313:	e8 48 10 ff ff       	callq  402360 <dcgettext@plt>
  411318:	4d 89 e1             	mov    %r12,%r9
  41131b:	48 89 c2             	mov    %rax,%rdx
  41131e:	4d 89 e8             	mov    %r13,%r8
  411321:	48 89 d9             	mov    %rbx,%rcx
  411324:	31 f6                	xor    %esi,%esi
  411326:	89 ef                	mov    %ebp,%edi
  411328:	31 c0                	xor    %eax,%eax
  41132a:	e8 41 14 ff ff       	callq  402770 <error@plt>
  41132f:	e8 ec 0e ff ff       	callq  402220 <abort@plt>
  411334:	be 10 6e 41 00       	mov    $0x416e10,%esi
  411339:	eb bf                	jmp    4112fa <__sprintf_chk@plt+0xea6a>
  41133b:	83 ff 04             	cmp    $0x4,%edi
  41133e:	be d1 6d 41 00       	mov    $0x416dd1,%esi
  411343:	74 b5                	je     4112fa <__sprintf_chk@plt+0xea6a>
  411345:	eb e8                	jmp    41132f <__sprintf_chk@plt+0xea9f>
  411347:	bb 09 6e 41 00       	mov    $0x416e09,%ebx
  41134c:	88 14 24             	mov    %dl,(%rsp)
  41134f:	c6 44 24 01 00       	movb   $0x0,0x1(%rsp)
  411354:	4c 29 d3             	sub    %r10,%rbx
  411357:	49 89 e5             	mov    %rsp,%r13
  41135a:	eb b0                	jmp    41130c <__sprintf_chk@plt+0xea7c>
  41135c:	0f 1f 40 00          	nopl   0x0(%rax)
  411360:	41 57                	push   %r15
  411362:	41 56                	push   %r14
  411364:	41 89 d6             	mov    %edx,%r14d
  411367:	41 55                	push   %r13
  411369:	41 54                	push   %r12
  41136b:	55                   	push   %rbp
  41136c:	53                   	push   %rbx
  41136d:	48 83 ec 28          	sub    $0x28,%rsp
  411371:	83 fa 24             	cmp    $0x24,%edx
  411374:	0f 87 0e 04 00 00    	ja     411788 <__sprintf_chk@plt+0xeef8>
  41137a:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
  41137f:	48 89 fd             	mov    %rdi,%rbp
  411382:	49 89 f7             	mov    %rsi,%r15
  411385:	48 85 f6             	test   %rsi,%rsi
  411388:	49 89 cd             	mov    %rcx,%r13
  41138b:	4d 89 c4             	mov    %r8,%r12
  41138e:	4c 0f 44 f8          	cmove  %rax,%r15
  411392:	0f b6 1f             	movzbl (%rdi),%ebx
  411395:	e8 e6 14 ff ff       	callq  402880 <__ctype_b_loc@plt>
  41139a:	48 8b 10             	mov    (%rax),%rdx
  41139d:	48 89 e8             	mov    %rbp,%rax
  4113a0:	eb 0d                	jmp    4113af <__sprintf_chk@plt+0xeb1f>
  4113a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4113a8:	48 83 c0 01          	add    $0x1,%rax
  4113ac:	0f b6 18             	movzbl (%rax),%ebx
  4113af:	44 0f b6 cb          	movzbl %bl,%r9d
  4113b3:	42 f6 44 4a 01 20    	testb  $0x20,0x1(%rdx,%r9,2)
  4113b9:	75 ed                	jne    4113a8 <__sprintf_chk@plt+0xeb18>
  4113bb:	80 fb 2d             	cmp    $0x2d,%bl
  4113be:	75 18                	jne    4113d8 <__sprintf_chk@plt+0xeb48>
  4113c0:	b8 04 00 00 00       	mov    $0x4,%eax
  4113c5:	48 83 c4 28          	add    $0x28,%rsp
  4113c9:	5b                   	pop    %rbx
  4113ca:	5d                   	pop    %rbp
  4113cb:	41 5c                	pop    %r12
  4113cd:	41 5d                	pop    %r13
  4113cf:	41 5e                	pop    %r14
  4113d1:	41 5f                	pop    %r15
  4113d3:	c3                   	retq   
  4113d4:	0f 1f 40 00          	nopl   0x0(%rax)
  4113d8:	e8 53 0e ff ff       	callq  402230 <__errno_location@plt>
  4113dd:	31 c9                	xor    %ecx,%ecx
  4113df:	44 89 f2             	mov    %r14d,%edx
  4113e2:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4113e8:	4c 89 fe             	mov    %r15,%rsi
  4113eb:	48 89 ef             	mov    %rbp,%rdi
  4113ee:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
  4113f3:	e8 68 10 ff ff       	callq  402460 <__strtoul_internal@plt>
  4113f8:	4d 8b 37             	mov    (%r15),%r14
  4113fb:	48 89 c3             	mov    %rax,%rbx
  4113fe:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
  411403:	49 39 ee             	cmp    %rbp,%r14
  411406:	0f 84 b3 00 00 00    	je     4114bf <__sprintf_chk@plt+0xec2f>
  41140c:	41 8b 00             	mov    (%r8),%eax
  41140f:	85 c0                	test   %eax,%eax
  411411:	75 2d                	jne    411440 <__sprintf_chk@plt+0xebb0>
  411413:	31 ed                	xor    %ebp,%ebp
  411415:	4d 85 e4             	test   %r12,%r12
  411418:	74 08                	je     411422 <__sprintf_chk@plt+0xeb92>
  41141a:	41 0f b6 16          	movzbl (%r14),%edx
  41141e:	84 d2                	test   %dl,%dl
  411420:	75 2e                	jne    411450 <__sprintf_chk@plt+0xebc0>
  411422:	49 89 5d 00          	mov    %rbx,0x0(%r13)
  411426:	48 83 c4 28          	add    $0x28,%rsp
  41142a:	89 e8                	mov    %ebp,%eax
  41142c:	5b                   	pop    %rbx
  41142d:	5d                   	pop    %rbp
  41142e:	41 5c                	pop    %r12
  411430:	41 5d                	pop    %r13
  411432:	41 5e                	pop    %r14
  411434:	41 5f                	pop    %r15
  411436:	c3                   	retq   
  411437:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  41143e:	00 00 
  411440:	83 f8 22             	cmp    $0x22,%eax
  411443:	bd 01 00 00 00       	mov    $0x1,%ebp
  411448:	0f 85 72 ff ff ff    	jne    4113c0 <__sprintf_chk@plt+0xeb30>
  41144e:	eb c5                	jmp    411415 <__sprintf_chk@plt+0xeb85>
  411450:	0f be f2             	movsbl %dl,%esi
  411453:	4c 89 e7             	mov    %r12,%rdi
  411456:	89 54 24 08          	mov    %edx,0x8(%rsp)
  41145a:	e8 71 0f ff ff       	callq  4023d0 <strchr@plt>
  41145f:	48 85 c0             	test   %rax,%rax
  411462:	8b 54 24 08          	mov    0x8(%rsp),%edx
  411466:	0f 84 94 00 00 00    	je     411500 <__sprintf_chk@plt+0xec70>
  41146c:	be 30 00 00 00       	mov    $0x30,%esi
  411471:	4c 89 e7             	mov    %r12,%rdi
  411474:	89 54 24 08          	mov    %edx,0x8(%rsp)
  411478:	e8 53 0f ff ff       	callq  4023d0 <strchr@plt>
  41147d:	48 85 c0             	test   %rax,%rax
  411480:	8b 54 24 08          	mov    0x8(%rsp),%edx
  411484:	74 1d                	je     4114a3 <__sprintf_chk@plt+0xec13>
  411486:	41 0f b6 46 01       	movzbl 0x1(%r14),%eax
  41148b:	3c 44                	cmp    $0x44,%al
  41148d:	0f 84 a7 02 00 00    	je     41173a <__sprintf_chk@plt+0xeeaa>
  411493:	3c 69                	cmp    $0x69,%al
  411495:	0f 84 87 02 00 00    	je     411722 <__sprintf_chk@plt+0xee92>
  41149b:	3c 42                	cmp    $0x42,%al
  41149d:	0f 84 97 02 00 00    	je     41173a <__sprintf_chk@plt+0xeeaa>
  4114a3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4114a8:	b8 00 04 00 00       	mov    $0x400,%eax
  4114ad:	83 ea 42             	sub    $0x42,%edx
  4114b0:	80 fa 35             	cmp    $0x35,%dl
  4114b3:	77 4b                	ja     411500 <__sprintf_chk@plt+0xec70>
  4114b5:	0f b6 d2             	movzbl %dl,%edx
  4114b8:	ff 24 d5 38 6e 41 00 	jmpq   *0x416e38(,%rdx,8)
  4114bf:	4d 85 e4             	test   %r12,%r12
  4114c2:	0f 84 f8 fe ff ff    	je     4113c0 <__sprintf_chk@plt+0xeb30>
  4114c8:	0f b6 55 00          	movzbl 0x0(%rbp),%edx
  4114cc:	84 d2                	test   %dl,%dl
  4114ce:	0f 84 ec fe ff ff    	je     4113c0 <__sprintf_chk@plt+0xeb30>
  4114d4:	0f be f2             	movsbl %dl,%esi
  4114d7:	4c 89 e7             	mov    %r12,%rdi
  4114da:	89 54 24 08          	mov    %edx,0x8(%rsp)
  4114de:	31 ed                	xor    %ebp,%ebp
  4114e0:	bb 01 00 00 00       	mov    $0x1,%ebx
  4114e5:	e8 e6 0e ff ff       	callq  4023d0 <strchr@plt>
  4114ea:	48 85 c0             	test   %rax,%rax
  4114ed:	8b 54 24 08          	mov    0x8(%rsp),%edx
  4114f1:	0f 85 75 ff ff ff    	jne    41146c <__sprintf_chk@plt+0xebdc>
  4114f7:	e9 c4 fe ff ff       	jmpq   4113c0 <__sprintf_chk@plt+0xeb30>
  4114fc:	0f 1f 40 00          	nopl   0x0(%rax)
  411500:	89 e8                	mov    %ebp,%eax
  411502:	49 89 5d 00          	mov    %rbx,0x0(%r13)
  411506:	83 c8 02             	or     $0x2,%eax
  411509:	e9 b7 fe ff ff       	jmpq   4113c5 <__sprintf_chk@plt+0xeb35>
  41150e:	31 d2                	xor    %edx,%edx
  411510:	09 d5                	or     %edx,%ebp
  411512:	48 63 c9             	movslq %ecx,%rcx
  411515:	49 01 ce             	add    %rcx,%r14
  411518:	89 e8                	mov    %ebp,%eax
  41151a:	83 c8 02             	or     $0x2,%eax
  41151d:	4d 89 37             	mov    %r14,(%r15)
  411520:	41 80 3e 00          	cmpb   $0x0,(%r14)
  411524:	0f 45 e8             	cmovne %eax,%ebp
  411527:	e9 f6 fe ff ff       	jmpq   411422 <__sprintf_chk@plt+0xeb92>
  41152c:	48 85 db             	test   %rbx,%rbx
  41152f:	0f 88 26 02 00 00    	js     41175b <__sprintf_chk@plt+0xeecb>
  411535:	48 01 db             	add    %rbx,%rbx
  411538:	31 d2                	xor    %edx,%edx
  41153a:	eb d4                	jmp    411510 <__sprintf_chk@plt+0xec80>
  41153c:	48 b8 ff ff ff ff ff 	movabs $0x7fffffffffffff,%rax
  411543:	ff 7f 00 
  411546:	48 39 c3             	cmp    %rax,%rbx
  411549:	0f 87 0c 02 00 00    	ja     41175b <__sprintf_chk@plt+0xeecb>
  41154f:	48 c1 e3 09          	shl    $0x9,%rbx
  411553:	31 d2                	xor    %edx,%edx
  411555:	eb b9                	jmp    411510 <__sprintf_chk@plt+0xec80>
  411557:	48 63 f8             	movslq %eax,%rdi
  41155a:	31 d2                	xor    %edx,%edx
  41155c:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  411563:	48 f7 f7             	div    %rdi
  411566:	be 07 00 00 00       	mov    $0x7,%esi
  41156b:	31 d2                	xor    %edx,%edx
  41156d:	eb 0f                	jmp    41157e <__sprintf_chk@plt+0xecee>
  41156f:	48 0f af df          	imul   %rdi,%rbx
  411573:	45 31 c0             	xor    %r8d,%r8d
  411576:	44 09 c2             	or     %r8d,%edx
  411579:	83 ee 01             	sub    $0x1,%esi
  41157c:	74 92                	je     411510 <__sprintf_chk@plt+0xec80>
  41157e:	48 39 d8             	cmp    %rbx,%rax
  411581:	73 ec                	jae    41156f <__sprintf_chk@plt+0xecdf>
  411583:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  41158a:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  411590:	eb e4                	jmp    411576 <__sprintf_chk@plt+0xece6>
  411592:	48 63 f8             	movslq %eax,%rdi
  411595:	31 d2                	xor    %edx,%edx
  411597:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  41159e:	48 f7 f7             	div    %rdi
  4115a1:	be 08 00 00 00       	mov    $0x8,%esi
  4115a6:	31 d2                	xor    %edx,%edx
  4115a8:	eb 13                	jmp    4115bd <__sprintf_chk@plt+0xed2d>
  4115aa:	48 0f af df          	imul   %rdi,%rbx
  4115ae:	45 31 c0             	xor    %r8d,%r8d
  4115b1:	44 09 c2             	or     %r8d,%edx
  4115b4:	83 ee 01             	sub    $0x1,%esi
  4115b7:	0f 84 53 ff ff ff    	je     411510 <__sprintf_chk@plt+0xec80>
  4115bd:	48 39 d8             	cmp    %rbx,%rax
  4115c0:	73 e8                	jae    4115aa <__sprintf_chk@plt+0xed1a>
  4115c2:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  4115c9:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4115cf:	eb e0                	jmp    4115b1 <__sprintf_chk@plt+0xed21>
  4115d1:	48 63 f0             	movslq %eax,%rsi
  4115d4:	31 d2                	xor    %edx,%edx
  4115d6:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4115dd:	48 f7 f6             	div    %rsi
  4115e0:	bf 04 00 00 00       	mov    $0x4,%edi
  4115e5:	31 d2                	xor    %edx,%edx
  4115e7:	48 39 d8             	cmp    %rbx,%rax
  4115ea:	0f 82 59 01 00 00    	jb     411749 <__sprintf_chk@plt+0xeeb9>
  4115f0:	45 31 c0             	xor    %r8d,%r8d
  4115f3:	48 0f af de          	imul   %rsi,%rbx
  4115f7:	44 09 c2             	or     %r8d,%edx
  4115fa:	83 ef 01             	sub    $0x1,%edi
  4115fd:	75 e8                	jne    4115e7 <__sprintf_chk@plt+0xed57>
  4115ff:	e9 0c ff ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  411604:	48 63 f0             	movslq %eax,%rsi
  411607:	31 d2                	xor    %edx,%edx
  411609:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  411610:	48 f7 f6             	div    %rsi
  411613:	bf 05 00 00 00       	mov    $0x5,%edi
  411618:	31 d2                	xor    %edx,%edx
  41161a:	eb 13                	jmp    41162f <__sprintf_chk@plt+0xed9f>
  41161c:	48 0f af de          	imul   %rsi,%rbx
  411620:	45 31 c0             	xor    %r8d,%r8d
  411623:	44 09 c2             	or     %r8d,%edx
  411626:	83 ef 01             	sub    $0x1,%edi
  411629:	0f 84 e1 fe ff ff    	je     411510 <__sprintf_chk@plt+0xec80>
  41162f:	48 39 d8             	cmp    %rbx,%rax
  411632:	73 e8                	jae    41161c <__sprintf_chk@plt+0xed8c>
  411634:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  41163b:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  411641:	eb e0                	jmp    411623 <__sprintf_chk@plt+0xed93>
  411643:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  41164a:	48 63 f8             	movslq %eax,%rdi
  41164d:	31 d2                	xor    %edx,%edx
  41164f:	48 89 f0             	mov    %rsi,%rax
  411652:	48 f7 f7             	div    %rdi
  411655:	48 39 c3             	cmp    %rax,%rbx
  411658:	0f 87 1d 01 00 00    	ja     41177b <__sprintf_chk@plt+0xeeeb>
  41165e:	48 0f af df          	imul   %rdi,%rbx
  411662:	48 39 d8             	cmp    %rbx,%rax
  411665:	0f 82 10 01 00 00    	jb     41177b <__sprintf_chk@plt+0xeeeb>
  41166b:	48 0f af df          	imul   %rdi,%rbx
  41166f:	31 d2                	xor    %edx,%edx
  411671:	e9 9a fe ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  411676:	48 c7 c6 ff ff ff ff 	mov    $0xffffffffffffffff,%rsi
  41167d:	48 63 f8             	movslq %eax,%rdi
  411680:	31 d2                	xor    %edx,%edx
  411682:	48 89 f0             	mov    %rsi,%rax
  411685:	48 f7 f7             	div    %rdi
  411688:	48 39 c3             	cmp    %rax,%rbx
  41168b:	76 de                	jbe    41166b <__sprintf_chk@plt+0xeddb>
  41168d:	48 89 f3             	mov    %rsi,%rbx
  411690:	ba 01 00 00 00       	mov    $0x1,%edx
  411695:	e9 76 fe ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  41169a:	48 63 f8             	movslq %eax,%rdi
  41169d:	31 d2                	xor    %edx,%edx
  41169f:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4116a6:	48 f7 f7             	div    %rdi
  4116a9:	be 06 00 00 00       	mov    $0x6,%esi
  4116ae:	31 d2                	xor    %edx,%edx
  4116b0:	eb 13                	jmp    4116c5 <__sprintf_chk@plt+0xee35>
  4116b2:	48 0f af df          	imul   %rdi,%rbx
  4116b6:	45 31 c0             	xor    %r8d,%r8d
  4116b9:	44 09 c2             	or     %r8d,%edx
  4116bc:	83 ee 01             	sub    $0x1,%esi
  4116bf:	0f 84 4b fe ff ff    	je     411510 <__sprintf_chk@plt+0xec80>
  4116c5:	48 39 d8             	cmp    %rbx,%rax
  4116c8:	73 e8                	jae    4116b2 <__sprintf_chk@plt+0xee22>
  4116ca:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  4116d1:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  4116d7:	eb e0                	jmp    4116b9 <__sprintf_chk@plt+0xee29>
  4116d9:	48 b8 ff ff ff ff ff 	movabs $0x3fffffffffffff,%rax
  4116e0:	ff 3f 00 
  4116e3:	48 39 c3             	cmp    %rax,%rbx
  4116e6:	77 73                	ja     41175b <__sprintf_chk@plt+0xeecb>
  4116e8:	48 c1 e3 0a          	shl    $0xa,%rbx
  4116ec:	31 d2                	xor    %edx,%edx
  4116ee:	e9 1d fe ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  4116f3:	48 63 f0             	movslq %eax,%rsi
  4116f6:	31 d2                	xor    %edx,%edx
  4116f8:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  4116ff:	48 f7 f6             	div    %rsi
  411702:	bf 03 00 00 00       	mov    $0x3,%edi
  411707:	31 d2                	xor    %edx,%edx
  411709:	48 39 d8             	cmp    %rbx,%rax
  41170c:	72 5e                	jb     41176c <__sprintf_chk@plt+0xeedc>
  41170e:	45 31 c0             	xor    %r8d,%r8d
  411711:	48 0f af de          	imul   %rsi,%rbx
  411715:	44 09 c2             	or     %r8d,%edx
  411718:	83 ef 01             	sub    $0x1,%edi
  41171b:	75 ec                	jne    411709 <__sprintf_chk@plt+0xee79>
  41171d:	e9 ee fd ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  411722:	31 c9                	xor    %ecx,%ecx
  411724:	41 80 7e 02 42       	cmpb   $0x42,0x2(%r14)
  411729:	b8 00 04 00 00       	mov    $0x400,%eax
  41172e:	0f 94 c1             	sete   %cl
  411731:	8d 4c 09 01          	lea    0x1(%rcx,%rcx,1),%ecx
  411735:	e9 73 fd ff ff       	jmpq   4114ad <__sprintf_chk@plt+0xec1d>
  41173a:	b9 02 00 00 00       	mov    $0x2,%ecx
  41173f:	b8 e8 03 00 00       	mov    $0x3e8,%eax
  411744:	e9 64 fd ff ff       	jmpq   4114ad <__sprintf_chk@plt+0xec1d>
  411749:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  411750:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  411756:	e9 9c fe ff ff       	jmpq   4115f7 <__sprintf_chk@plt+0xed67>
  41175b:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  411762:	ba 01 00 00 00       	mov    $0x1,%edx
  411767:	e9 a4 fd ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  41176c:	48 c7 c3 ff ff ff ff 	mov    $0xffffffffffffffff,%rbx
  411773:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  411779:	eb 9a                	jmp    411715 <__sprintf_chk@plt+0xee85>
  41177b:	ba 01 00 00 00       	mov    $0x1,%edx
  411780:	48 89 f3             	mov    %rsi,%rbx
  411783:	e9 88 fd ff ff       	jmpq   411510 <__sprintf_chk@plt+0xec80>
  411788:	b9 e8 6f 41 00       	mov    $0x416fe8,%ecx
  41178d:	ba 60 00 00 00       	mov    $0x60,%edx
  411792:	be e0 6b 41 00       	mov    $0x416be0,%esi
  411797:	bf f0 6b 41 00       	mov    $0x416bf0,%edi
  41179c:	e8 af 0c ff ff       	callq  402450 <__assert_fail@plt>
  4117a1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4117a8:	00 00 00 
  4117ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4117b0:	48 83 ec 08          	sub    $0x8,%rsp
  4117b4:	85 ff                	test   %edi,%edi
  4117b6:	74 48                	je     411800 <__sprintf_chk@plt+0xef70>
  4117b8:	83 ff 0a             	cmp    $0xa,%edi
  4117bb:	89 f8                	mov    %edi,%eax
  4117bd:	74 09                	je     4117c8 <__sprintf_chk@plt+0xef38>
  4117bf:	48 83 c4 08          	add    $0x8,%rsp
  4117c3:	c3                   	retq   
  4117c4:	0f 1f 40 00          	nopl   0x0(%rax)
  4117c8:	48 8b 16             	mov    (%rsi),%rdx
  4117cb:	bf b1 37 41 00       	mov    $0x4137b1,%edi
  4117d0:	b9 0a 00 00 00       	mov    $0xa,%ecx
  4117d5:	48 89 d6             	mov    %rdx,%rsi
  4117d8:	f3 a6                	repz cmpsb %es:(%rdi),%ds:(%rsi)
  4117da:	75 e3                	jne    4117bf <__sprintf_chk@plt+0xef2f>
  4117dc:	48 89 d7             	mov    %rdx,%rdi
  4117df:	e8 ec 0f ff ff       	callq  4027d0 <freecon@plt>
  4117e4:	e8 47 0a ff ff       	callq  402230 <__errno_location@plt>
  4117e9:	c7 00 3d 00 00 00    	movl   $0x3d,(%rax)
  4117ef:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  4117f4:	eb c9                	jmp    4117bf <__sprintf_chk@plt+0xef2f>
  4117f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4117fd:	00 00 00 
  411800:	e8 2b 0a ff ff       	callq  402230 <__errno_location@plt>
  411805:	c7 00 5f 00 00 00    	movl   $0x5f,(%rax)
  41180b:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  411810:	eb ad                	jmp    4117bf <__sprintf_chk@plt+0xef2f>
  411812:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
  411819:	1f 84 00 00 00 00 00 
  411820:	53                   	push   %rbx
  411821:	48 89 f3             	mov    %rsi,%rbx
  411824:	e8 b7 0d ff ff       	callq  4025e0 <getfilecon@plt>
  411829:	48 89 de             	mov    %rbx,%rsi
  41182c:	89 c7                	mov    %eax,%edi
  41182e:	5b                   	pop    %rbx
  41182f:	e9 7c ff ff ff       	jmpq   4117b0 <__sprintf_chk@plt+0xef20>
  411834:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  41183b:	00 00 00 00 00 
  411840:	53                   	push   %rbx
  411841:	48 89 f3             	mov    %rsi,%rbx
  411844:	e8 f7 0c ff ff       	callq  402540 <lgetfilecon@plt>
  411849:	48 89 de             	mov    %rbx,%rsi
  41184c:	89 c7                	mov    %eax,%edi
  41184e:	5b                   	pop    %rbx
  41184f:	e9 5c ff ff ff       	jmpq   4117b0 <__sprintf_chk@plt+0xef20>
  411854:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
  41185b:	00 00 00 00 00 
  411860:	53                   	push   %rbx
  411861:	48 89 f3             	mov    %rsi,%rbx
  411864:	e8 b7 0b ff ff       	callq  402420 <fgetfilecon@plt>
  411869:	48 89 de             	mov    %rbx,%rsi
  41186c:	89 c7                	mov    %eax,%edi
  41186e:	5b                   	pop    %rbx
  41186f:	e9 3c ff ff ff       	jmpq   4117b0 <__sprintf_chk@plt+0xef20>
  411874:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  41187b:	00 00 00 
  41187e:	66 90                	xchg   %ax,%ax
  411880:	41 54                	push   %r12
  411882:	55                   	push   %rbp
  411883:	48 89 fd             	mov    %rdi,%rbp
  411886:	53                   	push   %rbx
  411887:	e8 e4 09 ff ff       	callq  402270 <__fpending@plt>
  41188c:	8b 5d 00             	mov    0x0(%rbp),%ebx
  41188f:	48 89 ef             	mov    %rbp,%rdi
  411892:	49 89 c4             	mov    %rax,%r12
  411895:	e8 96 04 00 00       	callq  411d30 <__sprintf_chk@plt+0xf4a0>
  41189a:	83 e3 20             	and    $0x20,%ebx
  41189d:	85 c0                	test   %eax,%eax
  41189f:	0f 95 c2             	setne  %dl
  4118a2:	85 db                	test   %ebx,%ebx
  4118a4:	75 1a                	jne    4118c0 <__sprintf_chk@plt+0xf030>
  4118a6:	84 d2                	test   %dl,%dl
  4118a8:	74 0a                	je     4118b4 <__sprintf_chk@plt+0xf024>
  4118aa:	4d 85 e4             	test   %r12,%r12
  4118ad:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
  4118b2:	74 2c                	je     4118e0 <__sprintf_chk@plt+0xf050>
  4118b4:	89 d8                	mov    %ebx,%eax
  4118b6:	5b                   	pop    %rbx
  4118b7:	5d                   	pop    %rbp
  4118b8:	41 5c                	pop    %r12
  4118ba:	c3                   	retq   
  4118bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4118c0:	84 d2                	test   %dl,%dl
  4118c2:	bb ff ff ff ff       	mov    $0xffffffff,%ebx
  4118c7:	75 eb                	jne    4118b4 <__sprintf_chk@plt+0xf024>
  4118c9:	e8 62 09 ff ff       	callq  402230 <__errno_location@plt>
  4118ce:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  4118d4:	89 d8                	mov    %ebx,%eax
  4118d6:	5b                   	pop    %rbx
  4118d7:	5d                   	pop    %rbp
  4118d8:	41 5c                	pop    %r12
  4118da:	c3                   	retq   
  4118db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4118e0:	e8 4b 09 ff ff       	callq  402230 <__errno_location@plt>
  4118e5:	31 db                	xor    %ebx,%ebx
  4118e7:	83 38 09             	cmpl   $0x9,(%rax)
  4118ea:	0f 95 c3             	setne  %bl
  4118ed:	f7 db                	neg    %ebx
  4118ef:	89 d8                	mov    %ebx,%eax
  4118f1:	5b                   	pop    %rbx
  4118f2:	5d                   	pop    %rbp
  4118f3:	41 5c                	pop    %r12
  4118f5:	c3                   	retq   
  4118f6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4118fd:	00 00 00 
  411900:	41 57                	push   %r15
  411902:	bf 0e 00 00 00       	mov    $0xe,%edi
  411907:	41 56                	push   %r14
  411909:	41 55                	push   %r13
  41190b:	41 54                	push   %r12
  41190d:	55                   	push   %rbp
  41190e:	53                   	push   %rbx
  41190f:	48 81 ec a8 00 00 00 	sub    $0xa8,%rsp
  411916:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  41191d:	00 00 
  41191f:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
  411926:	00 
  411927:	31 c0                	xor    %eax,%eax
  411929:	e8 32 0d ff ff       	callq  402660 <nl_langinfo@plt>
  41192e:	4c 8b 35 23 9a 20 00 	mov    0x209a23(%rip),%r14        # 61b358 <stderr@@GLIBC_2.2.5+0xd08>
  411935:	48 85 c0             	test   %rax,%rax
  411938:	48 89 c3             	mov    %rax,%rbx
  41193b:	b8 19 69 41 00       	mov    $0x416919,%eax
  411940:	48 0f 44 d8          	cmove  %rax,%rbx
  411944:	4d 85 f6             	test   %r14,%r14
  411947:	75 21                	jne    41196a <__sprintf_chk@plt+0xf0da>
  411949:	e9 86 00 00 00       	jmpq   4119d4 <__sprintf_chk@plt+0xf144>
  41194e:	66 90                	xchg   %ax,%ax
  411950:	4c 89 f7             	mov    %r14,%rdi
  411953:	e8 28 0a ff ff       	callq  402380 <strlen@plt>
  411958:	49 8d 6c 06 01       	lea    0x1(%r14,%rax,1),%rbp
  41195d:	48 89 ef             	mov    %rbp,%rdi
  411960:	e8 1b 0a ff ff       	callq  402380 <strlen@plt>
  411965:	4c 8d 74 05 01       	lea    0x1(%rbp,%rax,1),%r14
  41196a:	41 0f b6 2e          	movzbl (%r14),%ebp
  41196e:	40 84 ed             	test   %bpl,%bpl
  411971:	74 29                	je     41199c <__sprintf_chk@plt+0xf10c>
  411973:	4c 89 f6             	mov    %r14,%rsi
  411976:	48 89 df             	mov    %rbx,%rdi
  411979:	e8 d2 0b ff ff       	callq  402550 <strcmp@plt>
  41197e:	85 c0                	test   %eax,%eax
  411980:	74 0d                	je     41198f <__sprintf_chk@plt+0xf0ff>
  411982:	40 80 fd 2a          	cmp    $0x2a,%bpl
  411986:	75 c8                	jne    411950 <__sprintf_chk@plt+0xf0c0>
  411988:	41 80 7e 01 00       	cmpb   $0x0,0x1(%r14)
  41198d:	75 c1                	jne    411950 <__sprintf_chk@plt+0xf0c0>
  41198f:	4c 89 f7             	mov    %r14,%rdi
  411992:	e8 e9 09 ff ff       	callq  402380 <strlen@plt>
  411997:	49 8d 5c 06 01       	lea    0x1(%r14,%rax,1),%rbx
  41199c:	80 3b 00             	cmpb   $0x0,(%rbx)
  41199f:	b8 fc 6f 41 00       	mov    $0x416ffc,%eax
  4119a4:	48 0f 44 d8          	cmove  %rax,%rbx
  4119a8:	48 8b 8c 24 98 00 00 	mov    0x98(%rsp),%rcx
  4119af:	00 
  4119b0:	64 48 33 0c 25 28 00 	xor    %fs:0x28,%rcx
  4119b7:	00 00 
  4119b9:	48 89 d8             	mov    %rbx,%rax
  4119bc:	0f 85 3b 03 00 00    	jne    411cfd <__sprintf_chk@plt+0xf46d>
  4119c2:	48 81 c4 a8 00 00 00 	add    $0xa8,%rsp
  4119c9:	5b                   	pop    %rbx
  4119ca:	5d                   	pop    %rbp
  4119cb:	41 5c                	pop    %r12
  4119cd:	41 5d                	pop    %r13
  4119cf:	41 5e                	pop    %r14
  4119d1:	41 5f                	pop    %r15
  4119d3:	c3                   	retq   
  4119d4:	bf 02 70 41 00       	mov    $0x417002,%edi
  4119d9:	e8 e2 07 ff ff       	callq  4021c0 <getenv@plt>
  4119de:	48 85 c0             	test   %rax,%rax
  4119e1:	49 89 c7             	mov    %rax,%r15
  4119e4:	74 09                	je     4119ef <__sprintf_chk@plt+0xf15f>
  4119e6:	80 38 00             	cmpb   $0x0,(%rax)
  4119e9:	0f 85 63 02 00 00    	jne    411c52 <__sprintf_chk@plt+0xf3c2>
  4119ef:	b8 07 00 00 00       	mov    $0x7,%eax
  4119f4:	41 bc 08 00 00 00    	mov    $0x8,%r12d
  4119fa:	41 bf f3 6f 41 00    	mov    $0x416ff3,%r15d
  411a00:	41 80 3c 07 2f       	cmpb   $0x2f,(%r15,%rax,1)
  411a05:	41 bd 01 00 00 00    	mov    $0x1,%r13d
  411a0b:	c7 44 24 08 01 00 00 	movl   $0x1,0x8(%rsp)
  411a12:	00 
  411a13:	0f 84 29 02 00 00    	je     411c42 <__sprintf_chk@plt+0xf3b2>
  411a19:	4d 01 e5             	add    %r12,%r13
  411a1c:	49 8d 7d 0e          	lea    0xe(%r13),%rdi
  411a20:	e8 1b 0c ff ff       	callq  402640 <malloc@plt>
  411a25:	48 85 c0             	test   %rax,%rax
  411a28:	48 89 c5             	mov    %rax,%rbp
  411a2b:	0f 84 c1 02 00 00    	je     411cf2 <__sprintf_chk@plt+0xf462>
  411a31:	4c 89 e2             	mov    %r12,%rdx
  411a34:	4c 89 fe             	mov    %r15,%rsi
  411a37:	48 89 c7             	mov    %rax,%rdi
  411a3a:	e8 81 0b ff ff       	callq  4025c0 <memcpy@plt>
  411a3f:	8b 54 24 08          	mov    0x8(%rsp),%edx
  411a43:	85 d2                	test   %edx,%edx
  411a45:	74 06                	je     411a4d <__sprintf_chk@plt+0xf1bd>
  411a47:	42 c6 44 25 00 2f    	movb   $0x2f,0x0(%rbp,%r12,1)
  411a4d:	49 01 ed             	add    %rbp,%r13
  411a50:	48 b8 63 68 61 72 73 	movabs $0x2e74657372616863,%rax
  411a57:	65 74 2e 
  411a5a:	be 00 00 02 00       	mov    $0x20000,%esi
  411a5f:	49 89 45 00          	mov    %rax,0x0(%r13)
  411a63:	b8 73 00 00 00       	mov    $0x73,%eax
  411a68:	41 c7 45 08 61 6c 69 	movl   $0x61696c61,0x8(%r13)
  411a6f:	61 
  411a70:	66 41 89 45 0c       	mov    %ax,0xc(%r13)
  411a75:	48 89 ef             	mov    %rbp,%rdi
  411a78:	31 c0                	xor    %eax,%eax
  411a7a:	e8 01 0d ff ff       	callq  402780 <open@plt>
  411a7f:	85 c0                	test   %eax,%eax
  411a81:	41 89 c4             	mov    %eax,%r12d
  411a84:	0f 88 9e 01 00 00    	js     411c28 <__sprintf_chk@plt+0xf398>
  411a8a:	be 21 3a 41 00       	mov    $0x413a21,%esi
  411a8f:	89 c7                	mov    %eax,%edi
  411a91:	e8 6a 0c ff ff       	callq  402700 <fdopen@plt>
  411a96:	48 85 c0             	test   %rax,%rax
  411a99:	49 89 c7             	mov    %rax,%r15
  411a9c:	0f 84 7e 01 00 00    	je     411c20 <__sprintf_chk@plt+0xf390>
  411aa2:	4c 8d 64 24 20       	lea    0x20(%rsp),%r12
  411aa7:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
  411aae:	00 00 
  411ab0:	49 8b 47 08          	mov    0x8(%r15),%rax
  411ab4:	49 3b 47 10          	cmp    0x10(%r15),%rax
  411ab8:	0f 83 43 01 00 00    	jae    411c01 <__sprintf_chk@plt+0xf371>
  411abe:	48 8d 50 01          	lea    0x1(%rax),%rdx
  411ac2:	49 89 57 08          	mov    %rdx,0x8(%r15)
  411ac6:	0f b6 38             	movzbl (%rax),%edi
  411ac9:	83 ff 20             	cmp    $0x20,%edi
  411acc:	74 e2                	je     411ab0 <__sprintf_chk@plt+0xf220>
  411ace:	8d 47 f7             	lea    -0x9(%rdi),%eax
  411ad1:	83 f8 01             	cmp    $0x1,%eax
  411ad4:	76 da                	jbe    411ab0 <__sprintf_chk@plt+0xf220>
  411ad6:	83 ff 23             	cmp    $0x23,%edi
  411ad9:	0f 84 96 01 00 00    	je     411c75 <__sprintf_chk@plt+0xf3e5>
  411adf:	4c 89 fe             	mov    %r15,%rsi
  411ae2:	e8 89 0b ff ff       	callq  402670 <ungetc@plt>
  411ae7:	48 8d 4c 24 60       	lea    0x60(%rsp),%rcx
  411aec:	31 c0                	xor    %eax,%eax
  411aee:	4c 89 e2             	mov    %r12,%rdx
  411af1:	be 12 70 41 00       	mov    $0x417012,%esi
  411af6:	4c 89 ff             	mov    %r15,%rdi
  411af9:	e8 a2 09 ff ff       	callq  4024a0 <fscanf@plt>
  411afe:	83 f8 01             	cmp    $0x1,%eax
  411b01:	0f 8e 95 01 00 00    	jle    411c9c <__sprintf_chk@plt+0xf40c>
  411b07:	4c 89 e2             	mov    %r12,%rdx
  411b0a:	8b 0a                	mov    (%rdx),%ecx
  411b0c:	48 83 c2 04          	add    $0x4,%rdx
  411b10:	8d 81 ff fe fe fe    	lea    -0x1010101(%rcx),%eax
  411b16:	f7 d1                	not    %ecx
  411b18:	21 c8                	and    %ecx,%eax
  411b1a:	25 80 80 80 80       	and    $0x80808080,%eax
  411b1f:	74 e9                	je     411b0a <__sprintf_chk@plt+0xf27a>
  411b21:	89 c1                	mov    %eax,%ecx
  411b23:	4c 8d 54 24 60       	lea    0x60(%rsp),%r10
  411b28:	c1 e9 10             	shr    $0x10,%ecx
  411b2b:	a9 80 80 00 00       	test   $0x8080,%eax
  411b30:	0f 44 c1             	cmove  %ecx,%eax
  411b33:	48 8d 4a 02          	lea    0x2(%rdx),%rcx
  411b37:	48 0f 44 d1          	cmove  %rcx,%rdx
  411b3b:	00 c0                	add    %al,%al
  411b3d:	48 83 da 03          	sbb    $0x3,%rdx
  411b41:	4c 29 e2             	sub    %r12,%rdx
  411b44:	41 8b 0a             	mov    (%r10),%ecx
  411b47:	49 83 c2 04          	add    $0x4,%r10
  411b4b:	8d 81 ff fe fe fe    	lea    -0x1010101(%rcx),%eax
  411b51:	f7 d1                	not    %ecx
  411b53:	21 c8                	and    %ecx,%eax
  411b55:	25 80 80 80 80       	and    $0x80808080,%eax
  411b5a:	74 e8                	je     411b44 <__sprintf_chk@plt+0xf2b4>
  411b5c:	89 c1                	mov    %eax,%ecx
  411b5e:	c1 e9 10             	shr    $0x10,%ecx
  411b61:	a9 80 80 00 00       	test   $0x8080,%eax
  411b66:	0f 44 c1             	cmove  %ecx,%eax
  411b69:	49 8d 4a 02          	lea    0x2(%r10),%rcx
  411b6d:	4c 0f 44 d1          	cmove  %rcx,%r10
  411b71:	00 c0                	add    %al,%al
  411b73:	48 8d 44 24 60       	lea    0x60(%rsp),%rax
  411b78:	49 83 da 03          	sbb    $0x3,%r10
  411b7c:	49 29 c2             	sub    %rax,%r10
  411b7f:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
  411b85:	49 8d 04 12          	lea    (%r10,%rdx,1),%rax
  411b89:	0f 85 2d 01 00 00    	jne    411cbc <__sprintf_chk@plt+0xf42c>
  411b8f:	48 8d 48 02          	lea    0x2(%rax),%rcx
  411b93:	48 8d 78 03          	lea    0x3(%rax),%rdi
  411b97:	48 89 54 24 18       	mov    %rdx,0x18(%rsp)
  411b9c:	4c 89 54 24 10       	mov    %r10,0x10(%rsp)
  411ba1:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  411ba6:	e8 95 0a ff ff       	callq  402640 <malloc@plt>
  411bab:	4c 8b 54 24 10       	mov    0x10(%rsp),%r10
  411bb0:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  411bb5:	49 89 c5             	mov    %rax,%r13
  411bb8:	4d 85 ed             	test   %r13,%r13
  411bbb:	0f 84 41 01 00 00    	je     411d02 <__sprintf_chk@plt+0xf472>
  411bc1:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
  411bc6:	48 c7 c7 fe ff ff ff 	mov    $0xfffffffffffffffe,%rdi
  411bcd:	4c 89 e6             	mov    %r12,%rsi
  411bd0:	48 29 d7             	sub    %rdx,%rdi
  411bd3:	4d 29 d6             	sub    %r10,%r14
  411bd6:	4c 01 f7             	add    %r14,%rdi
  411bd9:	4c 01 ef             	add    %r13,%rdi
  411bdc:	e8 7f 06 ff ff       	callq  402260 <strcpy@plt>
  411be1:	4b 8d 7c 35 ff       	lea    -0x1(%r13,%r14,1),%rdi
  411be6:	48 8d 74 24 60       	lea    0x60(%rsp),%rsi
  411beb:	4d 89 ee             	mov    %r13,%r14
  411bee:	e8 6d 06 ff ff       	callq  402260 <strcpy@plt>
  411bf3:	49 8b 47 08          	mov    0x8(%r15),%rax
  411bf7:	49 3b 47 10          	cmp    0x10(%r15),%rax
  411bfb:	0f 82 bd fe ff ff    	jb     411abe <__sprintf_chk@plt+0xf22e>
  411c01:	4c 89 ff             	mov    %r15,%rdi
  411c04:	e8 a7 05 ff ff       	callq  4021b0 <__uflow@plt>
  411c09:	83 f8 ff             	cmp    $0xffffffff,%eax
  411c0c:	89 c7                	mov    %eax,%edi
  411c0e:	0f 84 88 00 00 00    	je     411c9c <__sprintf_chk@plt+0xf40c>
  411c14:	e9 b0 fe ff ff       	jmpq   411ac9 <__sprintf_chk@plt+0xf239>
  411c19:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  411c20:	44 89 e7             	mov    %r12d,%edi
  411c23:	e8 98 08 ff ff       	callq  4024c0 <close@plt>
  411c28:	41 be 19 69 41 00    	mov    $0x416919,%r14d
  411c2e:	48 89 ef             	mov    %rbp,%rdi
  411c31:	e8 ba 05 ff ff       	callq  4021f0 <free@plt>
  411c36:	4c 89 35 1b 97 20 00 	mov    %r14,0x20971b(%rip)        # 61b358 <stderr@@GLIBC_2.2.5+0xd08>
  411c3d:	e9 28 fd ff ff       	jmpq   41196a <__sprintf_chk@plt+0xf0da>
  411c42:	45 31 ed             	xor    %r13d,%r13d
  411c45:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
  411c4c:	00 
  411c4d:	e9 c7 fd ff ff       	jmpq   411a19 <__sprintf_chk@plt+0xf189>
  411c52:	48 89 c7             	mov    %rax,%rdi
  411c55:	e8 26 07 ff ff       	callq  402380 <strlen@plt>
  411c5a:	48 85 c0             	test   %rax,%rax
  411c5d:	49 89 c4             	mov    %rax,%r12
  411c60:	74 e0                	je     411c42 <__sprintf_chk@plt+0xf3b2>
  411c62:	48 8d 40 ff          	lea    -0x1(%rax),%rax
  411c66:	e9 95 fd ff ff       	jmpq   411a00 <__sprintf_chk@plt+0xf170>
  411c6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  411c70:	83 f8 ff             	cmp    $0xffffffff,%eax
  411c73:	74 1e                	je     411c93 <__sprintf_chk@plt+0xf403>
  411c75:	49 8b 47 08          	mov    0x8(%r15),%rax
  411c79:	49 3b 47 10          	cmp    0x10(%r15),%rax
  411c7d:	0f 83 9a 00 00 00    	jae    411d1d <__sprintf_chk@plt+0xf48d>
  411c83:	48 8d 50 01          	lea    0x1(%rax),%rdx
  411c87:	49 89 57 08          	mov    %rdx,0x8(%r15)
  411c8b:	0f b6 00             	movzbl (%rax),%eax
  411c8e:	83 f8 0a             	cmp    $0xa,%eax
  411c91:	75 dd                	jne    411c70 <__sprintf_chk@plt+0xf3e0>
  411c93:	83 f8 ff             	cmp    $0xffffffff,%eax
  411c96:	0f 85 14 fe ff ff    	jne    411ab0 <__sprintf_chk@plt+0xf220>
  411c9c:	4c 89 ff             	mov    %r15,%rdi
  411c9f:	e8 8c 00 00 00       	callq  411d30 <__sprintf_chk@plt+0xf4a0>
  411ca4:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
  411ca9:	48 85 c0             	test   %rax,%rax
  411cac:	0f 84 76 ff ff ff    	je     411c28 <__sprintf_chk@plt+0xf398>
  411cb2:	41 c6 04 06 00       	movb   $0x0,(%r14,%rax,1)
  411cb7:	e9 72 ff ff ff       	jmpq   411c2e <__sprintf_chk@plt+0xf39e>
  411cbc:	48 03 44 24 08       	add    0x8(%rsp),%rax
  411cc1:	4c 89 f7             	mov    %r14,%rdi
  411cc4:	4c 89 54 24 18       	mov    %r10,0x18(%rsp)
  411cc9:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
  411cce:	48 8d 48 02          	lea    0x2(%rax),%rcx
  411cd2:	48 8d 70 03          	lea    0x3(%rax),%rsi
  411cd6:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
  411cdb:	e8 00 0a ff ff       	callq  4026e0 <realloc@plt>
  411ce0:	4c 8b 54 24 18       	mov    0x18(%rsp),%r10
  411ce5:	49 89 c5             	mov    %rax,%r13
  411ce8:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  411ced:	e9 c6 fe ff ff       	jmpq   411bb8 <__sprintf_chk@plt+0xf328>
  411cf2:	41 be 19 69 41 00    	mov    $0x416919,%r14d
  411cf8:	e9 39 ff ff ff       	jmpq   411c36 <__sprintf_chk@plt+0xf3a6>
  411cfd:	e8 9e 06 ff ff       	callq  4023a0 <__stack_chk_fail@plt>
  411d02:	4c 89 f7             	mov    %r14,%rdi
  411d05:	41 be 19 69 41 00    	mov    $0x416919,%r14d
  411d0b:	e8 e0 04 ff ff       	callq  4021f0 <free@plt>
  411d10:	4c 89 ff             	mov    %r15,%rdi
  411d13:	e8 18 00 00 00       	callq  411d30 <__sprintf_chk@plt+0xf4a0>
  411d18:	e9 11 ff ff ff       	jmpq   411c2e <__sprintf_chk@plt+0xf39e>
  411d1d:	4c 89 ff             	mov    %r15,%rdi
  411d20:	e8 8b 04 ff ff       	callq  4021b0 <__uflow@plt>
  411d25:	e9 64 ff ff ff       	jmpq   411c8e <__sprintf_chk@plt+0xf3fe>
  411d2a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  411d30:	41 54                	push   %r12
  411d32:	55                   	push   %rbp
  411d33:	53                   	push   %rbx
  411d34:	48 89 fb             	mov    %rdi,%rbx
  411d37:	e8 b4 08 ff ff       	callq  4025f0 <fileno@plt>
  411d3c:	85 c0                	test   %eax,%eax
  411d3e:	48 89 df             	mov    %rbx,%rdi
  411d41:	78 5c                	js     411d9f <__sprintf_chk@plt+0xf50f>
  411d43:	e8 68 09 ff ff       	callq  4026b0 <__freading@plt>
  411d48:	85 c0                	test   %eax,%eax
  411d4a:	75 34                	jne    411d80 <__sprintf_chk@plt+0xf4f0>
  411d4c:	48 89 df             	mov    %rbx,%rdi
  411d4f:	e8 5c 00 00 00       	callq  411db0 <__sprintf_chk@plt+0xf520>
  411d54:	85 c0                	test   %eax,%eax
  411d56:	74 44                	je     411d9c <__sprintf_chk@plt+0xf50c>
  411d58:	e8 d3 04 ff ff       	callq  402230 <__errno_location@plt>
  411d5d:	44 8b 20             	mov    (%rax),%r12d
  411d60:	48 89 df             	mov    %rbx,%rdi
  411d63:	48 89 c5             	mov    %rax,%rbp
  411d66:	e8 a5 05 ff ff       	callq  402310 <fclose@plt>
  411d6b:	45 85 e4             	test   %r12d,%r12d
  411d6e:	74 09                	je     411d79 <__sprintf_chk@plt+0xf4e9>
  411d70:	44 89 65 00          	mov    %r12d,0x0(%rbp)
  411d74:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  411d79:	5b                   	pop    %rbx
  411d7a:	5d                   	pop    %rbp
  411d7b:	41 5c                	pop    %r12
  411d7d:	c3                   	retq   
  411d7e:	66 90                	xchg   %ax,%ax
  411d80:	48 89 df             	mov    %rbx,%rdi
  411d83:	e8 68 08 ff ff       	callq  4025f0 <fileno@plt>
  411d88:	31 f6                	xor    %esi,%esi
  411d8a:	ba 01 00 00 00       	mov    $0x1,%edx
  411d8f:	89 c7                	mov    %eax,%edi
  411d91:	e8 9a 06 ff ff       	callq  402430 <lseek@plt>
  411d96:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  411d9a:	75 b0                	jne    411d4c <__sprintf_chk@plt+0xf4bc>
  411d9c:	48 89 df             	mov    %rbx,%rdi
  411d9f:	5b                   	pop    %rbx
  411da0:	5d                   	pop    %rbp
  411da1:	41 5c                	pop    %r12
  411da3:	e9 68 05 ff ff       	jmpq   402310 <fclose@plt>
  411da8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  411daf:	00 
  411db0:	48 85 ff             	test   %rdi,%rdi
  411db3:	53                   	push   %rbx
  411db4:	48 89 fb             	mov    %rdi,%rbx
  411db7:	74 09                	je     411dc2 <__sprintf_chk@plt+0xf532>
  411db9:	e8 f2 08 ff ff       	callq  4026b0 <__freading@plt>
  411dbe:	85 c0                	test   %eax,%eax
  411dc0:	75 0e                	jne    411dd0 <__sprintf_chk@plt+0xf540>
  411dc2:	48 89 df             	mov    %rbx,%rdi
  411dc5:	5b                   	pop    %rbx
  411dc6:	e9 85 08 ff ff       	jmpq   402650 <fflush@plt>
  411dcb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  411dd0:	f7 03 00 01 00 00    	testl  $0x100,(%rbx)
  411dd6:	74 ea                	je     411dc2 <__sprintf_chk@plt+0xf532>
  411dd8:	48 89 df             	mov    %rbx,%rdi
  411ddb:	ba 01 00 00 00       	mov    $0x1,%edx
  411de0:	31 f6                	xor    %esi,%esi
  411de2:	e8 09 00 00 00       	callq  411df0 <__sprintf_chk@plt+0xf560>
  411de7:	48 89 df             	mov    %rbx,%rdi
  411dea:	5b                   	pop    %rbx
  411deb:	e9 60 08 ff ff       	jmpq   402650 <fflush@plt>
  411df0:	53                   	push   %rbx
  411df1:	48 89 fb             	mov    %rdi,%rbx
  411df4:	48 83 ec 10          	sub    $0x10,%rsp
  411df8:	48 8b 47 08          	mov    0x8(%rdi),%rax
  411dfc:	48 39 47 10          	cmp    %rax,0x10(%rdi)
  411e00:	74 0e                	je     411e10 <__sprintf_chk@plt+0xf580>
  411e02:	48 83 c4 10          	add    $0x10,%rsp
  411e06:	48 89 df             	mov    %rbx,%rdi
  411e09:	5b                   	pop    %rbx
  411e0a:	e9 81 09 ff ff       	jmpq   402790 <fseeko@plt>
  411e0f:	90                   	nop
  411e10:	48 8b 47 20          	mov    0x20(%rdi),%rax
  411e14:	48 39 47 28          	cmp    %rax,0x28(%rdi)
  411e18:	75 e8                	jne    411e02 <__sprintf_chk@plt+0xf572>
  411e1a:	48 83 7f 48 00       	cmpq   $0x0,0x48(%rdi)
  411e1f:	75 e1                	jne    411e02 <__sprintf_chk@plt+0xf572>
  411e21:	89 54 24 0c          	mov    %edx,0xc(%rsp)
  411e25:	48 89 34 24          	mov    %rsi,(%rsp)
  411e29:	e8 c2 07 ff ff       	callq  4025f0 <fileno@plt>
  411e2e:	8b 54 24 0c          	mov    0xc(%rsp),%edx
  411e32:	48 8b 34 24          	mov    (%rsp),%rsi
  411e36:	89 c7                	mov    %eax,%edi
  411e38:	e8 f3 05 ff ff       	callq  402430 <lseek@plt>
  411e3d:	48 83 f8 ff          	cmp    $0xffffffffffffffff,%rax
  411e41:	74 0c                	je     411e4f <__sprintf_chk@plt+0xf5bf>
  411e43:	83 23 ef             	andl   $0xffffffef,(%rbx)
  411e46:	48 89 83 90 00 00 00 	mov    %rax,0x90(%rbx)
  411e4d:	31 c0                	xor    %eax,%eax
  411e4f:	48 83 c4 10          	add    $0x10,%rsp
  411e53:	5b                   	pop    %rbx
  411e54:	c3                   	retq   
  411e55:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  411e5c:	00 00 00 
  411e5f:	90                   	nop
  411e60:	41 57                	push   %r15
  411e62:	41 89 ff             	mov    %edi,%r15d
  411e65:	41 56                	push   %r14
  411e67:	49 89 f6             	mov    %rsi,%r14
  411e6a:	41 55                	push   %r13
  411e6c:	49 89 d5             	mov    %rdx,%r13
  411e6f:	41 54                	push   %r12
  411e71:	4c 8d 25 78 7f 20 00 	lea    0x207f78(%rip),%r12        # 619df0 <_fini@@Base+0x207ef4>
  411e78:	55                   	push   %rbp
  411e79:	48 8d 2d 78 7f 20 00 	lea    0x207f78(%rip),%rbp        # 619df8 <_fini@@Base+0x207efc>
  411e80:	53                   	push   %rbx
  411e81:	4c 29 e5             	sub    %r12,%rbp
  411e84:	31 db                	xor    %ebx,%ebx
  411e86:	48 c1 fd 03          	sar    $0x3,%rbp
  411e8a:	48 83 ec 08          	sub    $0x8,%rsp
  411e8e:	e8 d5 02 ff ff       	callq  402168 <_init@@Base>
  411e93:	48 85 ed             	test   %rbp,%rbp
  411e96:	74 1e                	je     411eb6 <__sprintf_chk@plt+0xf626>
  411e98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  411e9f:	00 
  411ea0:	4c 89 ea             	mov    %r13,%rdx
  411ea3:	4c 89 f6             	mov    %r14,%rsi
  411ea6:	44 89 ff             	mov    %r15d,%edi
  411ea9:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
  411ead:	48 83 c3 01          	add    $0x1,%rbx
  411eb1:	48 39 eb             	cmp    %rbp,%rbx
  411eb4:	75 ea                	jne    411ea0 <__sprintf_chk@plt+0xf610>
  411eb6:	48 83 c4 08          	add    $0x8,%rsp
  411eba:	5b                   	pop    %rbx
  411ebb:	5d                   	pop    %rbp
  411ebc:	41 5c                	pop    %r12
  411ebe:	41 5d                	pop    %r13
  411ec0:	41 5e                	pop    %r14
  411ec2:	41 5f                	pop    %r15
  411ec4:	c3                   	retq   
  411ec5:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  411ecc:	00 00 00 00 
  411ed0:	f3 c3                	repz retq 
  411ed2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  411ed9:	00 00 00 
  411edc:	0f 1f 40 00          	nopl   0x0(%rax)
  411ee0:	48 8d 05 c1 84 20 00 	lea    0x2084c1(%rip),%rax        # 61a3a8 <_fini@@Base+0x2084ac>
  411ee7:	48 85 c0             	test   %rax,%rax
  411eea:	74 0a                	je     411ef6 <__sprintf_chk@plt+0xf666>
  411eec:	48 8b 10             	mov    (%rax),%rdx
  411eef:	31 f6                	xor    %esi,%esi
  411ef1:	e9 ba 08 ff ff       	jmpq   4027b0 <__cxa_atexit@plt>
  411ef6:	31 d2                	xor    %edx,%edx
  411ef8:	eb f5                	jmp    411eef <__sprintf_chk@plt+0xf65f>

Disassembly of section .fini:

0000000000411efc <_fini@@Base>:
  411efc:	48 83 ec 08          	sub    $0x8,%rsp
  411f00:	48 83 c4 08          	add    $0x8,%rsp
  411f04:	c3                   	retq   
