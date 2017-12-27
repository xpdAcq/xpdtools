import operator as op

from streamz import Stream

from xpdtools.tools import (polarization_correction, generate_binner, mask_img,
                            integrate, z_score_image)


def conf_iq_pipeline(polarization=.99,
                     mask_kwargs=None,
                     bg_scale=1):
    if mask_kwargs is None:
        mask_kwargs = {}
    foreground = Stream(stream_name='foreground')

    background = Stream(stream_name='background')

    background_corrected_img = (
        foreground.
            zip_latest(background).
            map(lambda x: op.sub(x[0], bg_scale * x[1]),
                stream_name='background corrected img')
    )

    geometry = Stream(stream_name='geometry')

    pol_corrected_img = (
        background_corrected_img.
            zip_latest(geometry).
            map(lambda a: polarization_correction(
            a[0], a[1], polarization_factor=polarization),
                stream_name='corrected image'))

    mask = (pol_corrected_img.
        zip_latest(geometry).
        map(lambda a, **kwargs: mask_img(a[0], a[1], **kwargs),
            stream_name='mask', **mask_kwargs))

    binner = (mask.
        zip_latest(geometry).
        map(lambda x: generate_binner(x[1], mask=x[0])))

    img_binner = pol_corrected_img.zip_latest(binner)

    iq = img_binner.map(lambda x: integrate(x[0], x[1],
                                            statistic='median'),
                        stream_name='I(Q)')

    z_score = img_binner.map(lambda x: z_score_image(x[0], x[1]),
                             stream_name='z score')

    return foreground, background, geometry, iq, z_score, mask