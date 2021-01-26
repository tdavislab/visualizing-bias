// mike, lewis, noah, james, lucas, william, jacob, daniel, henry, matthew
// lisa, emma, sophia, emily, chloe, hannah, lily, claire, anna

// Fill the textboxes while testing
// let TESTING = false;
let TESTING = true;

// Initialize global variables
let LABEL_VISIBILITY = true;
let MEAN_VISIBILITY = true;
let EVAL_VISIBILITY = true;
let REMOVE_POINTS = false;
let ANIMSTEP_COUNTER = 0;
let ANIMATION_DURATION = 1000;
let AXIS_TOLERANCE = 0.1;

// Set global color-scale
let color = d3.scaleOrdinal(d3.schemeTableau10);
let shape = d3.scaleOrdinal([1, 2, 3, 4, 5, 6],
    [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolCross, d3.symbolStar].map(d => symbolGenerator(d)));

function symbolGenerator(symbolObj) {
    return d3.symbol().type(symbolObj).size(200)();
}

function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function process_response(response, eval = false, debiased = false) {
    let data = [];
    let vec1key = debiased ? 'debiased_vectors1' : 'vectors1';
    let vec2key = debiased ? 'debiased_vectors2' : 'vectors2';
    let debiased_veckey = debiased ? 'debiased_evalvecs' : 'evalvecs'

    for (let i = 0; i < response[vec1key].length; i++) {
        data.push({'position': response[vec1key][i], 'label': response['words1'][i], 'group': 1});
    }

    for (let i = 0; i < response[vec2key].length; i++) {
        data.push({'position': response[vec2key][i], 'label': response['words2'][i], 'group': 2});
    }

    if (eval) {
        for (let i = 0; i < response[debiased_veckey].length; i++) {
            data.push({'position': response[debiased_veckey][i], 'label': response.evalwords[i], 'group': 3});
        }
    }

    return data;
}

function vector_mean(arr) {
    let mean = [0, 0];
    let i;
    for (i = 0; i < arr.length; i++) {
        mean[0] += arr[i][0];
        mean[1] += arr[i][1];
    }
    mean[0] /= i;
    mean[1] /= i;
    return mean;
}

function check_if_mean(datapoint) {
    if (datapoint.label.startsWith('mean')) {
        return 1
    }
    return 0;
}

function remove_point() {
    let element = d3.select(this);
    let label = element.datum().label;
    let group = element.datum().group;
    if (group === 1) {
        let seedwords = $('#seedword-text-1').val().split(', ');
        let filtered = seedwords.filter(elem => elem !== label);
        $('#seedword-text-1').val(filtered.join(', '));
    }
    if (group === 2) {
        let seedwords = $('#seedword-text-2').val().split(', ');
        let filtered = seedwords.filter(elem => elem !== label);
        $('#seedword-text-2').val(filtered.join(', '));
    }
    $('#seedword-form-submit').click();
}

function draw_svg_scatter(parent_svg, response, plotTitle, debiased = false) {
    parent_svg.selectAll('*').remove();
    let margin = {top: 20, right: 20, bottom: 20, left: 40};
    let width = parent_svg.node().width.baseVal.value - margin.left - margin.right;
    let height = parent_svg.node().height.baseVal.value - margin.top - margin.bottom;
    // let data = process_response(response, eval, debiased);

    // if (mean) {
    //     let mean1 = vector_mean(response['vectors1']);
    //     let mean2 = vector_mean(response['vectors2']);
    //     data.push({'position': mean1, 'label': 'mean1', 'group': 1});
    //     data.push({'position': mean2, 'label': 'mean2', 'group': 2});
    // }

    // Append group to the svg
    let svg = parent_svg.append('g')
        .attr('id', plotTitle + 'group')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // set the ranges
    let x = d3.scaleLinear().range([0, width - 30]);
    let y = d3.scaleLinear().range([height, 0]);
    // x.domain(d3.extent(data, d => d.position[0])).nice();
    // y.domain(d3.extent(data, d => d.position[1])).nice();
    x.domain([response.bounds.xmin - AXIS_TOLERANCE, response.bounds.xmax + AXIS_TOLERANCE]).nice();
    y.domain([response.bounds.ymin - AXIS_TOLERANCE, response.bounds.ymax + AXIS_TOLERANCE]).nice();

    // If two-means then draw the line
    // if (mean) {
    //     let mean1 = data[data.length - 2];
    //     let mean2 = data[data.length - 1];
    //     svg.append('line')
    //         .attr('id', 'mean-line')
    //         .attr('stroke', 'black')
    //         .attr('stroke-width', 8)
    //         .attr('stroke-opacity', 0.5)
    //         .attr('x1', x(mean1.position[0]))
    //         .attr('y1', y(mean1.position[1]))
    //         .attr('x2', x(mean2.position[0]))
    //         .attr('y2', y(mean2.position[1]));
    // }
    let data = debiased ? response.debiased : response.base;

    // Add the scatterplot
    let datapoint_group = svg.selectAll('g')
        .data(data)
        .enter()
        .append('g')
        .attr('class', d => 'datapoint-group group-' + d.group)
        .attr('transform', d => 'translate(' + x(d.x) + ',' + y(d.y) + ')')
        .on('mouseover', function () {
            parent_svg.selectAll('g.datapoint-group').classed('translucent', true);
            d3.select(this).classed('translucent', false);
        })
        .on('mouseout', function () {
            parent_svg.selectAll('g.datapoint-group').classed('translucent', false);
        })

    // Class label
    datapoint_group.append('foreignObject')
        .attr('x', 15)
        .attr('y', -10)
        .attr('width', '1px')
        .attr('height', '1px')
        .attr('class', 'fobj')
        .append('xhtml:div')
        .attr('class', 'class-label')
        .html(d => d.label);

    // Remove buttons
    datapoint_group.append('text')
        .attr('class', 'fa cross-button')
        .attr('x', 10)
        .attr('y', -10)
        .attr('visibility', 'hidden')
        .on('click', remove_point)
        .text('\uf057');

    // datapoint_group.append('circle')
    //     .attr('r', 8)
    //     .attr('fill', d => color(d.group))
    //     .attr('stroke', 'black')
    //     .attr('stroke-width', d => check_if_mean(d) * 3)
    //     .attr('class', 'point');

    datapoint_group.append('path')
        .attr('fill', d => color(d.group))
        .attr('d', d => shape(d.group))


    // Add the X Axis
    svg.append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .classed('axis', true)
        .call(d3.axisBottom(x));

    // Add the Y Axis
    svg.append('g')
        .classed('axis', true)
        .call(d3.axisLeft(y));

    // d3.select('#play-control-play').on('click', function () {
    //     let new_data = process_response(response, eval, false);
    //
    //     svg.selectAll('.datapoint-group').data(new_data)
    //         .transition()
    //         .duration(5)
    //         .attr('transform', d => 'translate(' + x(d.position[0]) + ',' + y(d.position[1]) + ')')
    //         .on('end', function () {
    //             svg.selectAll('.datapoint-group').data(process_response(response, eval, true))
    //                 .transition()
    //                 .duration(5000)
    //                 .attr('transform', d => 'translate(' + x(d.position[0]) + ',' + y(d.position[1]) + ')');
    //         })
    // })
}

function draw_axes(svg, width, height, x, y) {
    // Add the X Axis
    svg.append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .classed('axis', true)
        .call(d3.axisBottom(x));

    // Add the Y Axis
    svg.append('g')
        .classed('axis', true)
        .call(d3.axisLeft(y));
}

function draw_scatter(svg, point_data, x, y) {
    // Add the scatterplot
    console.log('In draw scatter', point_data);

    let datapoint_group = svg.selectAll('g')
        .data(point_data)
        .enter()
        .append('g')
        .attr('class', d => 'datapoint-group group-' + d.group)
        .attr('transform', d => 'translate(' + x(d.x) + ',' + y(d.y) + ')')
        .on('mouseover', function () {
            svg.selectAll('g.datapoint-group').classed('translucent', true);
            d3.select(this).classed('translucent', false);
        })
        .on('mouseout', function () {
            svg.selectAll('g.datapoint-group').classed('translucent', false);
        })

    // Class label
    datapoint_group.append('foreignObject')
        .attr('x', 15)
        .attr('y', -10)
        .attr('width', '1px')
        .attr('height', '1px')
        .attr('class', 'fobj')
        .append('xhtml:div')
        .attr('class', 'class-label')
        .html(d => d.label);

    // Remove buttons
    datapoint_group.append('text')
        .attr('class', 'fa cross-button')
        .attr('x', 10)
        .attr('y', -10)
        .attr('visibility', 'hidden')
        .on('click', remove_point)
        .text('\uf057');

    // datapoint_group.append('circle')
    //     .attr('r', 8)
    //     .attr('fill', d => color(d.group))
    //     .attr('stroke', 'black')
    //     .attr('stroke-width', d => check_if_mean(d) * 3)
    //     .attr('class', 'point');
    datapoint_group.append('path')
        .attr('fill', d => color(d.group))
        .attr('d', d => shape(d.group))
}

function add_groups(data) {
    let grouped_data = [];

    for (let i = 0; i < data['vectors1'].length; i++) {
        grouped_data.push({'position': data['vectors1'][i], 'label': data['words1'][i], 'group': 1});
    }

    for (let i = 0; i < data['vectors2'].length; i++) {
        grouped_data.push({'position': data['vectors2'][i], 'label': data['words2'][i], 'group': 2});
    }

    for (let i = 0; i < data['evalvecs'].length; i++) {
        grouped_data.push({'position': data['evalvecs'][i], 'label': data['evalwords'][i], 'group': 3});
    }

    return grouped_data;
}

function setup_animation(anim_svg, response, identifier) {
    function update_anim_svg(svg, x_axis, y_axis, step) {
        let explanation_text = step <= response.explanations.length ? response.explanations[step] : 'No explanation found.';
        $('#explanation-text').text(explanation_text);

        svg.selectAll('g')
            .data(response.anim_steps[step])
            .transition()
            .duration(ANIMATION_DURATION)
            .attr('transform', d => 'translate(' + x_axis(d.x) + ',' + y_axis(d.y) + ')');
    }

    try {
        // console.log('setting up stuff');
        console.log(response);
        let margin = {top: 20, right: 20, bottom: 20, left: 40};
        let width = anim_svg.node().width.baseVal.value - margin.left - margin.right;
        let height = anim_svg.node().height.baseVal.value - margin.top - margin.bottom;

        // set the ranges
        let x_axis = d3.scaleLinear().range([0, width - 30]);
        let y_axis = d3.scaleLinear().range([height, 0]);
        x_axis.domain([response.bounds.xmin - AXIS_TOLERANCE, response.bounds.xmax + AXIS_TOLERANCE]).nice();
        y_axis.domain([response.bounds.ymin - AXIS_TOLERANCE, response.bounds.ymax + AXIS_TOLERANCE]).nice();

        let step_indicator = anim_svg.append('text')
            .attr('id', 'step-indicator')
            .attr('x', width)
            .attr('y', 20)
            .attr('text-anchor', 'end')
            .text('Step=0');

        let svg = anim_svg.append('g')
            .attr('id', identifier + 'group')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // let data = add_groups(response.anim_steps[0]);
        draw_scatter(svg, response.anim_steps[0], x_axis, y_axis);
        draw_axes(svg, width, height, x_axis, y_axis);
        $('#explanation-text').text(response.explanations[0]);

        // Step back
        let step_backward_btn = $('#play-control-sb');
        let step_forward_btn = $('#play-control-sf');
        let fast_backward_btn = $('#play-control-fb');
        let fast_forward_btn = $('#play-control-ff');

        btn_active(step_backward_btn, false);
        btn_active(fast_backward_btn, false);
        btn_active(step_forward_btn, true);
        btn_active(fast_forward_btn, true);

        // Step backward
        step_backward_btn.unbind('click').on('click', function (e) {
            // if already at 0, do nothing
            if (ANIMSTEP_COUNTER === 0) {
                console.log('Already at first step');
            } else {
                ANIMSTEP_COUNTER -= 1;
                btn_active(step_forward_btn, true);
                btn_active(fast_forward_btn, true);
                console.log(`Loading ANIMSTEP=${ANIMSTEP_COUNTER}`)

                update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER);

                if (ANIMSTEP_COUNTER === 0) {
                    btn_active(step_backward_btn, false);
                    btn_active(fast_backward_btn, false);
                }
            }
            d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);
        })

        // To the first step
        fast_backward_btn.unbind('click').on('click', function (e) {
            if (ANIMSTEP_COUNTER === 0) {
                console.log('Already at first step');
            } else {
                ANIMSTEP_COUNTER = 0;
                btn_active(step_forward_btn, true);
                btn_active(fast_forward_btn, true);
                console.log(`Loading ANIMSTEP=${ANIMSTEP_COUNTER}`)

                update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER);

                if (ANIMSTEP_COUNTER === 0) {
                    btn_active(step_backward_btn, false);
                    btn_active(fast_backward_btn, false);
                }
            }
            d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);

        })

        // Step forward
        step_forward_btn.unbind('click').on('click', function (e) {
            if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
                console.log('Already at last step');
            } else {
                ANIMSTEP_COUNTER += 1;
                btn_active(step_backward_btn, true);
                btn_active(fast_backward_btn, true);
                console.log(`Loading ANIMSTEP=${ANIMSTEP_COUNTER}`)

                update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER);

                if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
                    btn_active(step_forward_btn, false);
                    btn_active(fast_forward_btn, false);
                }
            }
            d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);
        })

        // To the last step
        fast_forward_btn.unbind('click').on('click', function (e) {
            if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
                console.log('Already at last step');
            } else {
                ANIMSTEP_COUNTER = response.anim_steps.length - 1;
                btn_active(step_backward_btn, true);
                btn_active(fast_backward_btn, true);
                console.log(`Loading ANIMSTEP=${ANIMSTEP_COUNTER}`)

                update_anim_svg(svg, x_axis, y_axis, ANIMSTEP_COUNTER);

                if (ANIMSTEP_COUNTER === response.anim_steps.length - 1) {
                    btn_active(step_forward_btn, false);
                    btn_active(fast_forward_btn, false);
                }
            }
            d3.select('#step-indicator').text(`Step=${ANIMSTEP_COUNTER}`);

        })
    } catch (e) {
        console.log(e);
    }
}

function btn_active(btn, bool_active) {
    btn.prop('disabled', !bool_active);
}

// Functionality for the dropdown-menus
$('#example-dropdown a').click(function (e) {
    $('#example-selection-button').text(this.innerHTML);
});

$('#algorithm-dropdown a').click(function (e) {
    let algorithm = this.innerHTML;
    $('#algorithm-selection-button').text('Algorithm: ' + algorithm);

    if (algorithm === 'Hard debiasing') {
        $('#equalize-holder').show();
    } else {
        $('#equalize-holder').hide();
    }

    if (algorithm === 'OSCaR') {
        $('#input-two-col-oscar').show();
    } else {
        $('#input-two-col-oscar').hide();
    }
});

$('#subspace-dropdown a').click(function (e) {
    try {
        let subspace_method = this.innerHTML;
        $('#subspace-selection-button').text('Subspace method: ' + subspace_method);

        if (subspace_method === 'PCA' || subspace_method === 'PCA-paired') {
            $('#seedword-text-2').hide();
        } else if (subspace_method === 'Two means' || subspace_method === 'Classification') {
            $('#seedword-text-2').show();
        } else {
            console.log('Incorrect subspace method');
        }
    } catch (e) {
        console.log(e);
    }
});

// Functionality for various toggle buttons
$('#toggle-labels-btn').click(function () {
    if (LABEL_VISIBILITY === true) {
        d3.selectAll('.fobj').attr('hidden', true);
        d3.select('#toggle-label-icon').attr('class', 'fa fa-toggle-on fa-rotate-180');
    } else {
        d3.selectAll('.fobj').attr('hidden', null);
        d3.select('#toggle-label-icon').attr('class', 'fa fa-toggle-on');
    }
    LABEL_VISIBILITY = !LABEL_VISIBILITY;
});

$('#toggle-eval-btn').click(function () {
    if (EVAL_VISIBILITY === true) {
        d3.selectAll('.group-3').attr('hidden', true);
        d3.select('#toggle-eval-icon').attr('class', 'fa fa-toggle-on fa-rotate-180');
    } else {
        d3.selectAll('.group-3').attr('hidden', null);
        d3.selectAll('#toggle-eval-icon').attr('class', 'fa fa-toggle-on');
    }
    EVAL_VISIBILITY = !EVAL_VISIBILITY;
});

$('#toggle-mean-btn').click(function () {
    if (MEAN_VISIBILITY === true) {
        d3.selectAll('#mean-line').attr('hidden', true);
        d3.select('#toggle-mean-icon').attr('class', 'fa fa-toggle-on fa-rotate-180');
    } else {
        d3.selectAll('#mean-line').attr('hidden', null);
        d3.selectAll('#toggle-mean-icon').attr('class', 'fa fa-toggle-on');
    }
    MEAN_VISIBILITY = !MEAN_VISIBILITY;
});

$('#remove-points-btn').click(function () {
    let cross_buttons = d3.selectAll('.cross-button')
    if (REMOVE_POINTS === true) {
        cross_buttons.attr('visibility', 'hidden');
    } else {
        cross_buttons.attr('visibility', 'visible');
        // .classed('shaker', !cross_buttons.classed('shaker'));
    }
    REMOVE_POINTS = !REMOVE_POINTS;
});

function svg_cleanup() {
    $('#pre-debiased-svg').empty();
    $('#animation-svg').empty();
    $('#post-debiased-svg').empty();
}

// Functionality for the 'Run' button
$('#seedword-form-submit').click(function () {
    try { // Perform cleanup
        svg_cleanup();
        ANIMSTEP_COUNTER = 0;

        let seedwords1 = $('#seedword-text-1').val();
        let seedwords2 = $('#seedword-text-2').val();
        let evalwords = $('#evaluation-list').val();
        let equalize = $('#equalize-list').val()
        let orth_subspace = $('#oscar-seedword-text-1').val();
        let algorithm = $('#algorithm-selection-button').text();
        let subspace_method = $('#subspace-selection-button').text();

        $.ajax({
            type: 'POST',
            url: '/seedwords2',
            data: {
                seedwords1: seedwords1, seedwords2: seedwords2, evalwords: evalwords, equalize: equalize, orth_subspace: orth_subspace,
                algorithm: algorithm, subspace_method: subspace_method
            },
            success: function (response) {
                // console.log(response);

                let predebiased_svg = d3.select('#pre-debiased-svg');
                draw_svg_scatter(predebiased_svg, response, 'Pre-debiasing', false,);

                let animation_svg = d3.select('#animation-svg');
                // draw_svg_scatter(animation_svg, response, 'Pre-debiasing', true, true);
                setup_animation(animation_svg, response, 'animation')

                let postdebiased_svg = d3.select('#post-debiased-svg');
                draw_svg_scatter(postdebiased_svg, response, 'Post-debiasing', true,);

                // $('#weat-predebiased').html('WEAT score = ' + response['weat_score_predebiased'].toFixed(3));
                // $('#weat-postdebiased').html('WEAT score = ' + response['weat_score_postdebiased'].toFixed(3));

                // enable toolbar buttons
                d3.select('#toggle-labels-btn').attr('disabled', null);
                d3.select('#remove-points-btn').attr('disabled', null);
                d3.select('#toggle-mean-btn').attr('disabled', null);
                d3.select('#toggle-eval-btn').attr('disabled', null);
                if (REMOVE_POINTS === true) {
                    REMOVE_POINTS = false;
                    $('#remove-points-btn').click();
                }

                if (LABEL_VISIBILITY === false) {
                    LABEL_VISIBILITY = true;
                    $('#toggle-labels-btn').click();
                }
            }
        });
    } catch (e) {
        console.log(e);
    }
});

if (TESTING) {
    try { // $('#seedword-text-1').val('mike, lewis, noah, james, lucas, william, jacob, daniel, henry, matthew');
        // $('#seedword-text-2').val('lisa, emma, sophia, emily, chloe, hannah, lily, claire, anna');
        $('#seedword-text-1').val('john, william, george, liam, andrew, michael, louis, tony, scott, jackson');
        $('#seedword-text-2').val('mary, victoria, carolina, maria, anne, kelly, marie, anna, sarah, jane');
        $('#evaluation-list').val('engineer, lawyer, mathematician, receptionist, homemaker, nurse, doctor');
        $('#equalize-list').val('monastery-convent, spokesman-spokeswoman, dad-mom, men-women, councilman-councilwoman,' +
            ' grandpa-grandma, grandsons-granddaughters, testosterone-estrogen, uncle-aunt, wives-husbands, father-mother,' +
            ' grandpa-grandma, he-she, boy-girl, boys-girls, brother-sister, brothers-sisters, businessman-businesswoman,' +
            ' chairman-chairwoman, colt-filly, congressman-congresswoman, dads-moms, dudes-gals, father-mother, fatherhood-motherhood,' +
            ' fathers-mothers, fella-granny, fraternity-sorority, gelding-mare, gentleman-lady, gentlemen-ladies,' +
            ' grandfather-grandmother, grandson-granddaughter, he-she, himself-herself, his-her, king-queen, kings-queens,' +
            ' male-female, males-females, man-woman, men-women, nephew-niece, prince-princess, schoolboy-schoolgirl, son-daughter, sons-daughters')
        $('#oscar-seedword-text-1').val('scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker')
        $('#algorithm-dropdown').children()[1].click();
        $('#subspace-dropdown-items').children()[1].click();
        $('#seedword-form-submit').click();
    } catch (e) {
        console.log(e);
    }
}
